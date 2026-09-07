/* gridanim.js — a small, dependency-free animation engine for the
 * "grid model of computation".
 *
 *   A 2-D lattice of byte cells at 1 µm pitch.  Cores (control units) sit at
 *   lattice sites with a small square footprint.  Serial tapes (input,
 *   output, instruction) sit at fixed sites.  Moving one byte along one
 *   micron of Manhattan path costs fJ_per_byte_um (default 1 fJ); signals fly
 *   at speed_um_per_ns (default c/160 = 1874 µm/ns); links serialise at
 *   link_bytes_per_ns (default 1 B/ns); a core issues one instruction per
 *   issue_ns (default 1 ns).  All four are parameters (config.constants).
 *
 * ES2017, no build step, no DOM access at load time, so the simulation core
 * (simulate / normalize / manhattan / fmt) can be `require`d under node.
 *
 * Public API
 *   GridAnim.create(containerEl, config) -> controller
 *   GridAnim.simulate(events, constants?, layout?) -> totals   (pure)
 *   GridAnim.normalize(events, constants?, layout?) -> resolved, time-sorted events
 *   GridAnim.manhattan([x,y],[x,y]) -> |dx|+|dy|
 *   GridAnim.fmt.energy(fJ) / fmt.time(ns) / fmt.int(n)
 *   GridAnim.DEFAULT_CONSTANTS, GridAnim.VERSION
 *
 * Event kinds (all have a start time t in ns; dur/energy default as noted):
 *   move    {t, from:[x,y], to:[x,y], bytes, label?, color?, energy?, dur?, route?:'xy'|'yx', core?}
 *           energy = bytes * manhattan * fJ_per_byte_um
 *           dur    = manhattan / speed_um_per_ns + bytes / link_bytes_per_ns
 *   sweep   {t, core, region, bytes, energy, dur, label?}
 *           energy defaults to 0 (caller supplies the aggregate), dur to bytes*issue_ns
 *   compute {t, core, dur, label?, energy?}      dur defaults to issue_ns, energy to 0
 *   io      {t, tape, bytes, energy, dur, label?} dur defaults to bytes/link_bytes_per_ns
 *   note    {t, text, dur?}                       dur defaults to 0 (shown until the next note)
 *
 * from/to may also be a core/tape id string when a layout is available; it
 * resolves to the centre cell of that footprint.  An optional `core` field
 * on any event attributes its energy and busy interval to that core.
 */
(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module && module.exports) module.exports = api;
  else root.GridAnim = api;
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var VERSION = '0.1.0';

  var DEFAULT_CONSTANTS = {
    fJ_per_byte_um: 1,      // energy to move one byte one micron of Manhattan path
    speed_um_per_ns: 1874,  // signal propagation, c/160
    link_bytes_per_ns: 1,   // serialisation rate of a link or tape
    issue_ns: 1             // one instruction per issue_ns
  };

  // ---------------------------------------------------------------- helpers
  function manhattan(a, b) { return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]); }
  function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
  function num(v, d) { var n = +v; return isFinite(n) ? n : d; }
  function sig3(v) { return String(Number(v.toPrecision(3))); }
  function scaleUnits(v, units) {
    var i = 0;
    while (i < units.length - 1 && v >= 999.5) { v /= 1000; i++; }
    return sig3(v) + ' ' + units[i];
  }

  var fmt = {
    /** fJ -> "4.56 pJ" (auto fJ/pJ/nJ/µJ/mJ/J, 3 significant digits). */
    energy: function (fJ) {
      if (!isFinite(fJ)) return String(fJ);
      if (fJ === 0) return '0 fJ';
      return (fJ < 0 ? '-' : '') + scaleUnits(Math.abs(fJ), ['fJ', 'pJ', 'nJ', 'µJ', 'mJ', 'J']);
    },
    /** ns -> "12.3 ns" (auto ps/ns/µs/ms/s). */
    time: function (ns) {
      if (!isFinite(ns)) return String(ns);
      if (ns === 0) return '0 ns';
      return (ns < 0 ? '-' : '') + scaleUnits(Math.abs(ns) * 1000, ['ps', 'ns', 'µs', 'ms', 's']);
    },
    /** 1024 -> "1,024" */
    int: function (n) { return String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ','); }
  };

  // Index a layout {cores, regions, tapes} by id.
  function indexLayout(layout) {
    var L = { cores: {}, regions: {}, tapes: {}, regionList: [], tapeList: [], coreList: [] };
    if (!layout) return L;
    (layout.cores || []).forEach(function (c) { L.cores[c.id] = c; L.coreList.push(c); });
    (layout.regions || []).forEach(function (r) { L.regions[r.id] = r; L.regionList.push(r); });
    (layout.tapes || []).forEach(function (t) { L.tapes[t.id] = t; L.tapeList.push(t); });
    return L;
  }
  function inRect(p, r) {
    return p[0] >= r.x && p[0] < r.x + (r.w || 1) && p[1] >= r.y && p[1] < r.y + (r.h || 1);
  }
  function resolvePoint(p, L, what) {
    if (Array.isArray(p)) return [num(p[0], 0), num(p[1], 0)];
    if (p && typeof p === 'object') return [num(p.x, 0), num(p.y, 0)];
    var o = L.cores[p] || L.tapes[p] || L.regions[p];
    if (o) return [Math.floor(o.x + (o.w || 1) / 2), Math.floor(o.y + (o.h || 1) / 2)];
    throw new Error('GridAnim: cannot resolve ' + what + ' = ' + JSON.stringify(p));
  }

  // ------------------------------------------------------------ normalize
  /** Resolve defaults (dur, energy, end, dist) and return a copy of the
   *  events sorted by start time (stable).  Pure. */
  function normalize(events, constants, layout) {
    var C = Object.assign({}, DEFAULT_CONSTANTS, constants || {});
    var L = indexLayout(layout);
    var out = [];
    for (var i = 0; i < (events || []).length; i++) {
      var e = events[i];
      if (!e) continue;
      var r = Object.assign({}, e);
      r.kind = e.kind || 'move';
      r.t = num(e.t, 0);
      r.seq = i;
      r.bytes = num(e.bytes, 0);
      switch (r.kind) {
        case 'move':
          r.from = resolvePoint(e.from, L, 'from');
          r.to = resolvePoint(e.to, L, 'to');
          r.dist = manhattan(r.from, r.to);
          r.dur = e.dur != null ? num(e.dur, 0)
                                : r.dist / C.speed_um_per_ns + r.bytes / C.link_bytes_per_ns;
          r.energy = e.energy != null ? num(e.energy, 0) : r.bytes * r.dist * C.fJ_per_byte_um;
          break;
        case 'sweep':
          r.dur = e.dur != null ? num(e.dur, 0) : r.bytes * C.issue_ns;
          r.energy = num(e.energy, 0);
          break;
        case 'compute':
          r.dur = e.dur != null ? num(e.dur, 0) : C.issue_ns;
          r.energy = num(e.energy, 0);
          break;
        case 'io':
          r.dur = e.dur != null ? num(e.dur, 0) : r.bytes / C.link_bytes_per_ns;
          r.energy = num(e.energy, 0);
          break;
        case 'note':
          r.dur = e.dur != null ? num(e.dur, 0) : 0;
          r.energy = 0;
          break;
        default:
          throw new Error('GridAnim: unknown event kind "' + r.kind + '" at index ' + i);
      }
      r.end = r.t + r.dur;
      out.push(r);
    }
    out.sort(function (a, b) { return (a.t - b.t) || (a.seq - b.seq); });
    return out;
  }

  // ------------------------------------------------------------- simulate
  /** Pure totals for a schedule.  makespan = max(t + dur); energy = sum of
   *  event energies; bytes_moved / byte_um over move events; per_core keyed
   *  by the events' `core` field (busy_ns = union of their [t, end]
   *  intervals); area = origin-anchored bounding square of every touched
   *  coordinate (move endpoints, `at`, and — when a layout is given — the
   *  footprints of referenced cores/regions/tapes): side = max(|x|,|y|). */
  function simulate(events, constants, layout) {
    var L = indexLayout(layout);
    var ev = normalize(events, constants, layout);
    var tot = { energy_fJ: 0, makespan_ns: 0, bytes_moved: 0, byte_um: 0, bytes_io: 0,
                events: ev.length, per_core: {}, side_um: 0, area_um2: 0 };
    var side = 0, cores = {};
    function touch(x, y) { var m = Math.max(Math.abs(x), Math.abs(y)); if (m > side) side = m; }
    function touchRect(r) { touch(r.x, r.y); touch(r.x + (r.w || 1) - 1, r.y + (r.h || 1) - 1); }
    for (var i = 0; i < ev.length; i++) {
      var e = ev[i];
      tot.energy_fJ += e.energy;
      if (e.end > tot.makespan_ns) tot.makespan_ns = e.end;
      if (e.kind === 'move') {
        tot.bytes_moved += e.bytes;
        tot.byte_um += e.bytes * e.dist;
        touch(e.from[0], e.from[1]);
        touch(e.to[0], e.to[1]);
      } else if (e.kind === 'io') {
        tot.bytes_io += e.bytes;
      }
      if (e.region != null && L.regions[e.region]) touchRect(L.regions[e.region]);
      if (e.core != null && L.cores[e.core]) touchRect(L.cores[e.core]);
      if (e.tape != null && L.tapes[e.tape]) touch(L.tapes[e.tape].x, L.tapes[e.tape].y);
      if (e.at) touch(num(e.at[0], 0), num(e.at[1], 0));
      if (e.core != null && e.kind !== 'note') {
        var pc = cores[e.core] || (cores[e.core] = { energy: 0, iv: [], n: 0 });
        pc.energy += e.energy; pc.n++; pc.iv.push([e.t, e.end]);
      }
    }
    Object.keys(cores).forEach(function (id) {
      var pc = cores[id];
      pc.iv.sort(function (a, b) { return a[0] - b[0]; });
      var busy = 0, cs = -Infinity, ce = -Infinity;
      for (var k = 0; k < pc.iv.length; k++) {
        var iv = pc.iv[k];
        if (iv[0] > ce) { if (ce > cs) busy += ce - cs; cs = iv[0]; ce = iv[1]; }
        else if (iv[1] > ce) ce = iv[1];
      }
      if (ce > cs) busy += ce - cs;
      tot.per_core[id] = { busy_ns: busy, energy_fJ: pc.energy, events: pc.n };
    });
    tot.side_um = side;
    tot.area_um2 = side * side;
    return tot;
  }

  // ---------------------------------------------------------------- theme
  var FALLBACK_LIGHT = { bg: '#fcfcfb', plane: '#f9f9f7', ink: '#0b0b0b', ink2: '#52514e',
    muted: '#898781', grid: '#e1e0d9', line: '#c3c2b7', accent: '#2a78d6', codeBg: '#f2f1ed',
    good: '#006300', bad: '#a33a3a' };
  var FALLBACK_DARK = { bg: '#1a1a19', plane: '#0d0d0d', ink: '#ffffff', ink2: '#c3c2b7',
    muted: '#898781', grid: '#2c2c2a', line: '#383835', accent: '#3987e5', codeBg: '#242422',
    good: '#0ca30c', bad: '#e66767' };
  // Tape colours are fixed hues chosen to read on both themes.
  var TAPE_COLORS = { 'in': '#2f9bd8', 'out': '#2f9e62', 'instr': '#d99a2b' };
  var TAPE_NAMES = { 'in': 'input', 'out': 'output', 'instr': 'instruction' };

  function readTheme() {
    var dark = typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: dark)').matches;
    var fb = dark ? FALLBACK_DARK : FALLBACK_LIGHT;
    var cs = typeof getComputedStyle === 'function' ? getComputedStyle(document.documentElement) : null;
    var th = { dark: dark, tape: TAPE_COLORS };
    Object.keys(fb).forEach(function (k) {
      var name = '--' + k.replace(/[A-Z]/g, function (c) { return '-' + c.toLowerCase(); });
      var v = cs ? cs.getPropertyValue(name).trim() : '';
      th[k] = v || fb[k];
    });
    return th;
  }
  // '#rgb' | '#rrggbb' | 'rgb(...)' -> 'rgba(r,g,b,a)'; anything else is returned as is.
  function alpha(col, a) {
    var m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(col);
    if (m) {
      var h = m[1];
      if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
      var n = parseInt(h, 16);
      return 'rgba(' + (n >> 16) + ',' + ((n >> 8) & 255) + ',' + (n & 255) + ',' + a + ')';
    }
    m = /^rgba?\(([^)]+)\)$/.exec(col);
    if (m) { var p = m[1].split(',').slice(0, 3).join(','); return 'rgba(' + p + ',' + a + ')'; }
    return col;
  }

  var CSS = [
    '.ga{font:13px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink,#0b0b0b);max-width:100%}',
    '.ga-title{font-weight:600;font-size:14px;margin:0 0 4px}',
    '.ga-stage{position:relative;background:var(--plane,#f9f9f7);border:1px solid var(--grid,#e1e0d9);border-radius:8px;overflow:hidden}',
    '.ga-stage canvas{display:block;width:100%}',
    '.ga-note{position:absolute;left:8px;bottom:4px;max-width:calc(100% - 16px);padding:2px 9px;border-radius:5px;background:var(--code-bg,#f2f1ed);color:var(--ink,#0b0b0b);border:1px solid var(--grid,#e1e0d9);font-size:12px;pointer-events:none;opacity:0;transition:opacity .2s}',
    '.ga-note.on{opacity:1}',
    '.ga-caption{color:var(--ink2,#52514e);font-size:12px;margin:5px 0 0}',
    '.ga-legend{display:flex;flex-wrap:wrap;gap:3px 14px;margin:6px 0 0;font-size:12px;color:var(--ink2,#52514e)}',
    '.ga-legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}',
    '.ga-controls{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;margin:8px 0 0}',
    '.ga-controls button{font:inherit;color:var(--ink,#0b0b0b);background:var(--code-bg,#f2f1ed);border:1px solid var(--line,#c3c2b7);border-radius:5px;padding:3px 10px;cursor:pointer;min-width:60px}',
    '.ga-controls button:hover{border-color:var(--accent,#2a78d6)}',
    '.ga-controls label{display:flex;align-items:center;gap:6px;color:var(--ink2,#52514e);white-space:nowrap;font-size:12px}',
    '.ga-controls input[type=range]{accent-color:var(--accent,#2a78d6);margin:0}',
    '.ga-speed{width:110px}',
    '.ga-scrub{flex-basis:100%;width:100%}',
    '.ga-readout{font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:12px;color:var(--ink2,#52514e);white-space:pre;flex-basis:100%}'
  ].join('\n');

  function injectCSS() {
    if (document.getElementById('gridanim-css')) return;
    var s = document.createElement('style');
    s.id = 'gridanim-css';
    s.textContent = CSS;
    document.head.appendChild(s);
  }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  // First index whose start time is > t (events sorted by t).
  function upperBound(ev, t) {
    var lo = 0, hi = ev.length;
    while (lo < hi) { var mid = (lo + hi) >> 1; if (ev[mid].t <= t) lo = mid + 1; else hi = mid; }
    return lo;
  }
  function niceStep(span, target) {
    var raw = span / target, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
    return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
  }

  // --------------------------------------------------------------- create
  function create(container, config) {
    if (typeof document === 'undefined') throw new Error('GridAnim.create needs a DOM');
    if (typeof container === 'string') container = document.querySelector(container);
    if (!container) throw new Error('GridAnim.create: container not found');
    config = config || {};
    injectCSS();

    var C = Object.assign({}, DEFAULT_CONSTANTS, config.constants || {});
    var world = Object.assign({ xmin: 0, xmax: 64, ymin: 0, ymax: 64 }, config.world || {});
    var layout = { cores: config.cores || [], regions: config.regions || [], tapes: config.tapes || [] };
    var L = indexLayout(layout);
    var maxDots = config.maxDots || 400;
    var maxHeight = config.maxHeight || 520;
    var theme = readTheme();
    var reduced = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

    // ---- DOM
    var rootEl = el('div', 'ga');
    if (config.title) rootEl.appendChild(el('div', 'ga-title', config.title));
    var stage = el('div', 'ga-stage');
    var canvas = document.createElement('canvas');
    var noteEl = el('div', 'ga-note');
    stage.appendChild(canvas); stage.appendChild(noteEl); rootEl.appendChild(stage);
    if (config.caption) rootEl.appendChild(el('div', 'ga-caption', config.caption));
    var legendEl = el('div', 'ga-legend'); rootEl.appendChild(legendEl);
    var ctl = el('div', 'ga-controls');
    var bPlay = el('button', null, 'Play'), bStep = el('button', null, 'Step'), bRestart = el('button', null, 'Restart');
    var speedLab = el('label'), speedIn = document.createElement('input'), speedTxt = el('span');
    speedIn.type = 'range'; speedIn.min = '0'; speedIn.max = '9'; speedIn.step = '0.05'; speedIn.className = 'ga-speed';
    speedLab.appendChild(document.createTextNode('speed')); speedLab.appendChild(speedIn); speedLab.appendChild(speedTxt);
    var scrub = document.createElement('input');
    scrub.type = 'range'; scrub.min = '0'; scrub.max = '1000'; scrub.step = '1'; scrub.value = '0'; scrub.className = 'ga-scrub';
    scrub.setAttribute('aria-label', 'scrub machine time');
    var readout = el('div', 'ga-readout');
    [bPlay, bStep, bRestart, speedLab, scrub, readout].forEach(function (x) { ctl.appendChild(x); });
    rootEl.appendChild(ctl);
    container.appendChild(rootEl);
    var ctx = canvas.getContext('2d');

    // ---- state
    var st = { ev: [], notes: [], n: 0, makespan: 0, tau: 0, cursor: 0, active: [], doneEnergy: 0,
               playing: false, speed: 1000, speedExplicit: config.speed != null, raf: 0, lastMs: 0,
               scrubbing: false, showLabels: config.labels !== false };
    var listeners = {};
    var view = null, stat = null, W = 0, H = 0, dpr = 1, destroyed = false, hasNoteBand = false;

    function emit(name, arg) {
      var ls = listeners[name]; if (!ls) return;
      for (var i = 0; i < ls.length; i++) { try { ls[i](arg); } catch (err) { console.error(err); } }
    }

    // ---- geometry: world <-> pixels (y up)
    function fit() {
      W = Math.max(120, container.clientWidth || rootEl.clientWidth || 600);
      var ml = 44, mr = 14, mt = 12, mb = 30 + (st.notes.length ? 24 : 0);   // note band below the ticks
      hasNoteBand = st.notes.length > 0;
      var ww = world.xmax - world.xmin, wh = world.ymax - world.ymin;
      var plotW = W - ml - mr;
      var plotH = Math.min(maxHeight - mt - mb, plotW * wh / ww);
      var s = Math.min(plotW / ww, plotH / wh);
      H = Math.round(mt + wh * s + mb);
      var ox = ml + (plotW - ww * s) / 2, oy = mt + wh * s;
      view = { s: s, ml: ml, mt: mt, ox: ox, oy: oy, pw: ww * s, ph: wh * s,
               x: function (wx) { return ox + (wx - world.xmin) * s; },
               y: function (wy) { return oy - (wy - world.ymin) * s; } };
      dpr = Math.min(2, (typeof devicePixelRatio === 'number' && devicePixelRatio) || 1);
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
      canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
      buildStatic();
    }
    function vx(x) { return view.x(x); }
    function vy(y) { return view.y(y); }
    function coreColor(c) { return (c && c.color) || theme.accent; }

    // ---- static layer: plane, grid, regions, cores, tapes, axes, origin
    function buildStatic() {
      var v = view, s = v.s;
      stat = document.createElement('canvas');
      stat.width = canvas.width; stat.height = canvas.height;
      var g = stat.getContext('2d');
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.fillStyle = theme.plane; g.fillRect(0, 0, W, H);
      g.fillStyle = theme.bg; g.fillRect(v.ox, v.mt, v.pw, v.ph);
      var ww = world.xmax - world.xmin;
      var cells = config.showCells === true || (config.showCells !== false && ww <= 160);
      g.lineWidth = 1;
      if (cells) {
        g.strokeStyle = alpha(theme.grid, 0.8); g.beginPath();
        for (var x = Math.ceil(world.xmin); x <= world.xmax; x++) {
          var px = Math.round(vx(x)) + 0.5; g.moveTo(px, v.mt); g.lineTo(px, v.mt + v.ph);
        }
        for (var y = Math.ceil(world.ymin); y <= world.ymax; y++) {
          var py = Math.round(vy(y)) + 0.5; g.moveTo(v.ox, py); g.lineTo(v.ox + v.pw, py);
        }
        g.stroke();
      }
      // regions (faint filled rects)
      g.font = '10px system-ui, sans-serif'; g.textBaseline = 'top'; g.textAlign = 'left';
      L.regionList.forEach(function (r) {
        var col = r.color || coreColor(L.cores[r.owner]);
        var x0 = vx(r.x), y0 = vy(r.y + r.h), w = r.w * s, h = r.h * s;
        g.fillStyle = alpha(col, 0.10); g.fillRect(x0, y0, w, h);
        g.strokeStyle = alpha(col, 0.45); g.strokeRect(Math.round(x0) + 0.5, Math.round(y0) + 0.5, Math.round(w), Math.round(h));
        if (r.label && w >= 44 && h >= 24) { g.fillStyle = alpha(theme.ink2, 0.8); g.fillText(r.label, x0 + 3, y0 + 2); }
      });
      // cores (filled squares with the id in the middle)
      L.coreList.forEach(function (c) {
        var w = (c.w || 1) * s, h = (c.h || 1) * s, x0 = vx(c.x), y0 = vy(c.y + (c.h || 1));
        var col = coreColor(c);
        g.fillStyle = alpha(col, 0.9); g.fillRect(x0, y0, w, h);
        g.strokeStyle = alpha(theme.ink, 0.5); g.strokeRect(Math.round(x0) + 0.5, Math.round(y0) + 0.5, Math.round(w), Math.round(h));
        var m = Math.min(w, h), above = c.label && c.label !== c.id ? String(c.label) : '';
        if (m >= 11) {                                // id fits inside the square
          g.fillStyle = '#fff'; g.textAlign = 'center'; g.textBaseline = 'middle';
          g.font = 'bold ' + Math.max(8, Math.min(13, m * 0.42)) + 'px system-ui, sans-serif';
          g.fillText(String(c.id), x0 + w / 2, y0 + h / 2 + 0.5);
        } else {                                      // too small: put the id above it
          above = above ? c.id + ' ' + above : String(c.id);
        }
        if (above) {
          g.fillStyle = theme.ink2; g.font = '10px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'bottom';
          g.fillText(above, x0 + w / 2, y0 - 2);
        }
      });
      // tapes: small labelled bars, fixed pixel size so they stay visible at any scale
      L.tapeList.forEach(function (t) {
        var col = theme.tape[t.kind] || theme.muted;
        var cx = vx(t.x + 0.5), cy = vy(t.y + 0.5), bw = 30, bh = 9;
        g.fillStyle = col; g.fillRect(cx - bw / 2, cy - bh / 2, bw, bh);
        g.strokeStyle = alpha(theme.ink, 0.5); g.strokeRect(Math.round(cx - bw / 2) + 0.5, Math.round(cy - bh / 2) + 0.5, bw, bh);
        g.fillStyle = theme.ink2; g.font = '10px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'bottom';
        g.fillText(t.label || t.id, cx, cy - bh / 2 - 2);
      });
      // axes ticks (µm) and origin mark
      g.fillStyle = theme.muted; g.strokeStyle = theme.line; g.font = '10px system-ui, sans-serif';
      var stepX = niceStep(ww, Math.max(3, Math.min(9, v.pw / 70))), stepY = niceStep(world.ymax - world.ymin, Math.max(3, v.ph / 45));
      g.textAlign = 'center'; g.textBaseline = 'top'; g.beginPath();
      for (var tx = Math.ceil(world.xmin / stepX) * stepX; tx <= world.xmax + 1e-9; tx += stepX) {
        var qx = Math.round(vx(tx)) + 0.5; g.moveTo(qx, v.mt + v.ph); g.lineTo(qx, v.mt + v.ph + 4);
        g.fillText(String(Math.round(tx * 1000) / 1000), qx, v.mt + v.ph + 6);
      }
      g.textAlign = 'right'; g.textBaseline = 'middle';
      for (var ty = Math.ceil(world.ymin / stepY) * stepY; ty <= world.ymax + 1e-9; ty += stepY) {
        var qy = Math.round(vy(ty)) + 0.5; g.moveTo(v.ox, qy); g.lineTo(v.ox - 4, qy);
        g.fillText(String(Math.round(ty * 1000) / 1000), v.ox - 7, qy);
      }
      g.stroke();
      g.textAlign = 'left'; g.textBaseline = 'top'; g.fillStyle = theme.muted;
      g.fillText('µm', v.ox + v.pw - 16, v.mt + v.ph + 14);
      if (0 >= world.xmin && 0 <= world.xmax && 0 >= world.ymin && 0 <= world.ymax) {
        var o0x = vx(0), o0y = vy(0);
        g.strokeStyle = theme.ink; g.lineWidth = 1.2; g.beginPath();
        g.moveTo(o0x - 5, o0y); g.lineTo(o0x + 5, o0y); g.moveTo(o0x, o0y - 5); g.lineTo(o0x, o0y + 5); g.stroke();
        g.beginPath(); g.arc(o0x, o0y, 2.2, 0, Math.PI * 2); g.fillStyle = theme.ink; g.fill();
      }
      g.strokeStyle = theme.line; g.lineWidth = 1;
      g.strokeRect(Math.round(v.ox) + 0.5, Math.round(v.mt) + 0.5, Math.round(v.pw), Math.round(v.ph));
    }

    // ---- schedule loading
    function tapeAt(p) {
      for (var i = 0; i < L.tapeList.length; i++) {
        var t = L.tapeList[i];
        if (t.x === p[0] && t.y === p[1]) return t;
      }
      return null;
    }
    function prepMove(e) {
      var a = [e.from[0] + 0.5, e.from[1] + 0.5], b = [e.to[0] + 0.5, e.to[1] + 0.5];
      var c = e.route === 'yx' ? [a[0], b[1]] : [b[0], a[1]];
      e._a = a; e._b = b; e._c = c;
      e._L1 = manhattan(a, c); e._L2 = manhattan(c, b); e._L = e._L1 + e._L2;
      e._pc = e._L > 0 ? e._L1 / e._L : 1;
      e._r = Math.max(4, Math.min(12, 2 + 0.9 * Math.log2(e.bytes + 1)));   // 4-byte messages stay visible at die scale
      if (e.color) { e._cls = 'custom'; return; }
      var tp = tapeAt(e.from) || tapeAt(e.to);
      if (tp) { e._cls = 'tape'; e._tk = tp.kind; return; }
      for (var i = 0; i < L.regionList.length; i++) {
        var r = L.regionList[i];
        if (inRect(e.from, r) && inRect(e.to, r)) { e._cls = 'fetch'; return; }
      }
      e._cls = 'msg';
    }
    function colorOf(e) {
      if (e.color) return e.color;
      if (e._cls === 'tape') return theme.tape[e._tk] || theme.muted;
      if (e._cls === 'fetch') return theme.ink2;
      if (e.kind === 'io') return theme.tape[(L.tapes[e.tape] || {}).kind] || theme.muted;
      return theme.accent;
    }
    function load(events) {
      st.ev = normalize(events || [], C, layout);
      st.notes = [];
      st.n = st.ev.length; st.makespan = 0;
      st.attnStart = []; st.attnEndMax = []; st.ff = false;
      var mxEnd = -Infinity;
      for (var i = 0; i < st.n; i++) {
        var e = st.ev[i];
        if (e.end > st.makespan) st.makespan = e.end;
        if (e.kind === 'move') prepMove(e);
        else if (e.kind === 'note') st.notes.push(e);
        if (e.kind === 'move' || e.kind === 'io') {   // "attention" intervals: something visibly moves
          st.attnStart.push(e.t); if (e.end > mxEnd) mxEnd = e.end; st.attnEndMax.push(mxEnd);
        }
      }
      if (!st.speedExplicit) {
        st.speed = st.makespan > 0 ? Math.pow(10, Math.round(Math.log10(st.makespan / 8))) : 1000;
        st.speed = Math.max(1, Math.min(1e9, st.speed));
      }
      speedIn.value = String(Math.log10(st.speed));
      if (view && hasNoteBand !== (st.notes.length > 0)) fit();
      buildLegend();
      rebuild(0);
      st.playing = false; st.lastMs = 0;
      updateButtons(); updateReadout(); schedule();
      emit('load', simulate(st.ev, C, layout));
    }
    function buildLegend() {
      legendEl.innerHTML = '';
      var items = config.legend;
      if (!items) {
        items = [];
        var seen = {};
        st.ev.forEach(function (e) {
          if (e.kind === 'move' && !seen[e._cls]) { seen[e._cls] = true; }
          if (e.kind === 'io') seen.io = true;
        });
        if (seen.msg) items.push({ color: theme.accent, label: 'message between cores' });
        if (seen.fetch) items.push({ color: theme.ink2, label: 'operand fetch within a region' });
        L.tapeList.forEach(function (t) {
          if (!seen['tape-' + t.kind]) { seen['tape-' + t.kind] = true; items.push({ color: theme.tape[t.kind], label: (TAPE_NAMES[t.kind] || t.kind) + ' tape' }); }
        });
      }
      items.forEach(function (it) {
        var sp = el('span'), sw = el('i'); sw.style.background = it.color;
        sp.appendChild(sw); sp.appendChild(document.createTextNode(it.label)); legendEl.appendChild(sp);
      });
      legendEl.style.display = items.length ? '' : 'none';
    }

    // ---- active set maintenance (per-frame work ∝ active events)
    function glowNs() { return 0.35 * st.speed; }   // arrival afterglow, in sim time
    function trailNs() { return 0.5 * st.speed; }   // trail behind a moving dot, in sim time
    function rebuild(tau) {                          // O(n); used for backward seeks
      st.cursor = upperBound(st.ev, tau);
      st.active = []; st.doneEnergy = 0;
      var keepUntil = tau - glowNs();
      for (var i = 0; i < st.cursor; i++) {
        var e = st.ev[i];
        if (e.end >= keepUntil) st.active.push(e); else st.doneEnergy += e.energy;
      }
      st.tau = tau;
    }
    function setTime(tau) {
      tau = Math.max(0, Math.min(st.makespan, tau));
      if (tau < st.tau) { rebuild(tau); return; }
      var ev = st.ev;
      while (st.cursor < st.n && ev[st.cursor].t <= tau) st.active.push(ev[st.cursor++]);
      var keepUntil = tau - glowNs(), keep = [];
      for (var i = 0; i < st.active.length; i++) {
        var e = st.active[i];
        if (e.end >= keepUntil) keep.push(e); else st.doneEnergy += e.energy;
      }
      st.active = keep;
      st.tau = tau;
    }
    function frac(e, tau) { return e.dur > 0 ? clamp01((tau - e.t) / e.dur) : (tau >= e.t ? 1 : 0); }
    function energyNow() {
      var E = st.doneEnergy;
      for (var i = 0; i < st.active.length; i++) E += st.active[i].energy * frac(st.active[i], st.tau);
      return E;
    }
    function currentNote() {
      var ns = st.notes, i = upperBound(ns, st.tau) - 1;
      if (i < 0) return null;
      var n = ns[i];
      if (n.dur > 0 && st.tau > n.end) return null;
      return n;
    }

    // ---- dynamic drawing
    function pathPoint(e, p) {
      if (e._L === 0) return e._b;
      var u = p * e._L;
      if (u <= e._L1) { var f = e._L1 ? u / e._L1 : 1; return [e._a[0] + (e._c[0] - e._a[0]) * f, e._a[1] + (e._c[1] - e._a[1]) * f]; }
      var f2 = e._L2 ? (u - e._L1) / e._L2 : 1;
      return [e._c[0] + (e._b[0] - e._c[0]) * f2, e._c[1] + (e._b[1] - e._c[1]) * f2];
    }
    function strokePath(g, e, pa, pb) {
      var A = pathPoint(e, pa), B = pathPoint(e, pb);
      g.beginPath(); g.moveTo(vx(A[0]), vy(A[1]));
      if (pa < e._pc && pb > e._pc) g.lineTo(vx(e._c[0]), vy(e._c[1]));
      g.lineTo(vx(B[0]), vy(B[1])); g.stroke();
    }
    function drawMove(g, e, tau, labelsOk) {
      var th = Math.min(tau, e.end);
      var p = e.dur > 0 ? clamp01((th - e.t) / e.dur) : 1;
      var p0 = e.dur > 0 ? clamp01((th - trailNs() - e.t) / e.dur) : 0;
      var fade = tau > e.end ? 1 - clamp01((tau - e.end) / glowNs()) : 1;
      if (fade <= 0) return;
      var col = colorOf(e), r = e._r;
      if (p > p0 && e._L > 0) {                       // trail: 6 segments of rising alpha
        g.strokeStyle = col; g.lineCap = 'butt'; g.lineJoin = 'round'; g.lineWidth = Math.max(1.5, r * 1.1);
        var N = 6;
        for (var k = 0; k < N; k++) {
          g.globalAlpha = fade * 0.5 * (k + 1) / N;
          strokePath(g, e, p0 + (p - p0) * k / N, p0 + (p - p0) * (k + 1) / N);
        }
      }
      var pt = pathPoint(e, p), cx = vx(pt[0]), cy = vy(pt[1]);
      g.globalAlpha = fade; g.fillStyle = col;
      g.beginPath(); g.arc(cx, cy, r, 0, Math.PI * 2); g.fill();
      if (tau >= e.end) {                             // arrival ring
        g.globalAlpha = fade * 0.7; g.strokeStyle = col; g.lineWidth = 1.5;
        g.beginPath(); g.arc(cx, cy, r + (1 - fade) * r * 2.5, 0, Math.PI * 2); g.stroke();
      }
      if (labelsOk && e.label && tau < e.end) {         // label rides with the dot, dropped on arrival
        g.globalAlpha = fade; g.fillStyle = theme.ink; g.font = '10px system-ui, sans-serif';
        g.textAlign = 'left'; g.textBaseline = 'bottom'; g.fillText(e.label, cx + r + 2, cy - 2);
      }
      g.globalAlpha = 1;
    }
    function drawBusy(g, core, col, fade, ms) {
      if (!core) return;
      var w = (core.w || 1) * view.s, h = (core.h || 1) * view.s, x0 = vx(core.x), y0 = vy(core.y + (core.h || 1));
      var pulse = 0.65 + 0.35 * Math.sin(ms / 160);
      g.globalAlpha = fade * pulse; g.strokeStyle = col; g.lineWidth = 3;
      g.strokeRect(x0 - 2, y0 - 2, w + 4, h + 4);
      g.globalAlpha = 1;
    }
    function drawSweep(g, e, tau, ms) {
      var r = L.regions[e.region], core = L.cores[e.core];
      var p = frac(e, tau), fade = tau > e.end ? 1 - clamp01((tau - e.end) / glowNs()) : 1;
      if (fade <= 0) return;
      var col = e.color || coreColor(core);
      if (r) {
        var x0 = vx(r.x), y0 = vy(r.y + r.h), w = r.w * view.s, h = r.h * view.s;
        g.globalAlpha = 0.3 * fade; g.fillStyle = col; g.fillRect(x0, y0, w * p, h);
        if (tau < e.end) {
          g.globalAlpha = 0.9 * fade; g.strokeStyle = col; g.lineWidth = 2;
          g.beginPath(); g.moveTo(x0 + w * p, y0); g.lineTo(x0 + w * p, y0 + h); g.stroke();
        }
        if (st.showLabels && e.label && w >= 40) {
          g.globalAlpha = fade; g.fillStyle = theme.ink; g.font = '10px system-ui, sans-serif';
          g.textAlign = 'left'; g.textBaseline = 'bottom'; g.fillText(e.label, x0 + 2, y0 + h - 2);
        }
        g.globalAlpha = 1;
      }
      drawBusy(g, core, col, fade, ms);
    }
    function drawCompute(g, e, tau, ms) {
      var core = L.cores[e.core];
      var fade = tau > e.end ? 1 - clamp01((tau - e.end) / glowNs()) : 1;
      if (fade <= 0 || !core) return;
      drawBusy(g, core, e.color || coreColor(core), fade, ms);
      if (st.showLabels && e.label) {
        g.globalAlpha = fade; g.fillStyle = theme.ink; g.font = '10px system-ui, sans-serif';
        g.textAlign = 'center'; g.textBaseline = 'top';
        g.fillText(e.label, vx(core.x + (core.w || 1) / 2), vy(core.y) + 4); g.globalAlpha = 1;
      }
    }
    function drawIo(g, e, tau) {
      var t = L.tapes[e.tape]; if (!t) return;
      var p = frac(e, tau), fade = tau > e.end ? 1 - clamp01((tau - e.end) / glowNs()) : 1;
      if (fade <= 0) return;
      var col = colorOf(e), cx = vx(t.x + 0.5), cy = vy(t.y + 0.5);
      g.globalAlpha = fade * 0.9; g.fillStyle = col; g.fillRect(cx - 15, cy - 6, 30 * (tau < e.end ? p : 1), 12);
      var ring = tau < e.end ? p : 1;
      g.globalAlpha = fade * (1 - ring) * 0.8 + (tau >= e.end ? 0 : 0.1); g.strokeStyle = col; g.lineWidth = 2;
      g.beginPath(); g.arc(cx, cy, 8 + 16 * ring, 0, Math.PI * 2); g.stroke();
      if (st.showLabels && e.label) {
        g.globalAlpha = fade; g.fillStyle = theme.ink; g.font = '10px system-ui, sans-serif';
        g.textAlign = 'center'; g.textBaseline = 'top'; g.fillText(e.label, cx, cy + 9);
      }
      g.globalAlpha = 1;
    }
    function draw(ms) {
      if (!view) return;
      var g = ctx, tau = st.tau;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, W, H);
      g.drawImage(stat, 0, 0, W, H);
      g.save();
      g.beginPath(); g.rect(view.ox - 1, view.mt - 1, view.pw + 2, view.ph + 2); g.clip();
      var act = st.active, i, e, dots = 0, hidden = 0, moves = 0;
      for (i = 0; i < act.length; i++) {
        e = act[i];
        if (e.kind === 'sweep') drawSweep(g, e, tau, ms);
        else if (e.kind === 'compute') drawCompute(g, e, tau, ms);
        else if (e.kind === 'io') drawIo(g, e, tau);
        else if (e.kind === 'move') moves++;
      }
      var labelsOk = st.showLabels && moves <= 40;
      for (i = 0; i < act.length; i++) {
        e = act[i];
        if (e.kind !== 'move') continue;
        if (dots < maxDots) { drawMove(g, e, tau, labelsOk); dots++; } else hidden++;
      }
      g.restore();
      if (hidden > 0) {
        var txt = '+' + fmt.int(hidden) + ' more in flight';
        g.font = '11px system-ui, sans-serif'; g.textAlign = 'right'; g.textBaseline = 'top';
        var tw = g.measureText(txt).width;
        g.fillStyle = alpha(theme.codeBg, 0.92); g.fillRect(view.ox + view.pw - tw - 14, view.mt + 4, tw + 10, 17);
        g.fillStyle = theme.ink; g.fillText(txt, view.ox + view.pw - 9, view.mt + 6);
      }
    }

    // ---- controls
    function updateButtons() {
      bPlay.textContent = st.playing ? 'Pause' : (st.tau >= st.makespan && st.makespan > 0 ? 'Replay' : 'Play');
    }
    function updateReadout() {
      readout.textContent = 't = ' + fmt.time(st.tau) + '   E = ' + fmt.energy(energyNow()) +
        '   events ' + fmt.int(st.cursor) + '/' + fmt.int(st.n) +
        (st.ff && st.playing ? '   \u25b6\u25b6 \u00d7' + fmt.int(config.fastForward === false ? 1 : Math.max(1, num(config.fastForward, 30))) + ' through local compute' : '');
      speedTxt.textContent = fmt.time(st.speed) + ' of machine time per second';
      if (!st.scrubbing) scrub.value = String(st.makespan > 0 ? Math.round(1000 * st.tau / st.makespan) : 0);
      var n = currentNote();
      if (n) { noteEl.textContent = n.text || ''; noteEl.classList.add('on'); } else noteEl.classList.remove('on');
    }
    function tickInfo() {
      return { t: st.tau, energy_fJ: energyNow(), started: st.cursor, total: st.n,
               playing: st.playing, speed: st.speed, makespan_ns: st.makespan };
    }
    /** Playback multiplier while nothing is visibly moving (only sweeps/computes are active):
     *  the long local phases are fast-forwarded by config.fastForward (default 30; false = off). */
    function idleFactor(tau) {
      var FF = config.fastForward === false ? 1 : Math.max(1, num(config.fastForward, 30));
      if (FF <= 1 || !st.attnStart || !st.attnStart.length) return { f: 1, next: st.makespan };
      var lo = 0, hi = st.attnStart.length;           // first start > tau
      while (lo < hi) { var mid = (lo + hi) >> 1; if (st.attnStart[mid] > tau) hi = mid; else lo = mid + 1; }
      if (lo > 0 && st.attnEndMax[lo - 1] > tau) return { f: 1, next: st.makespan };
      return { f: FF, next: lo < st.attnStart.length ? st.attnStart[lo] : st.makespan };
    }
    function frame(ms) {
      st.raf = 0;
      if (destroyed) return;
      if (st.playing) {
        var dt = st.lastMs ? (ms - st.lastMs) / 1000 : 0;
        st.lastMs = ms;
        if (dt > 0.25) dt = 0.25;                     // tab was hidden: do not jump
        var ff = idleFactor(st.tau);
        var nt = st.tau + dt * st.speed * ff.f;
        if (ff.f > 1 && nt > ff.next) nt = ff.next;   // never skip past the next visible move
        st.ff = ff.f > 1;
        if (nt >= st.makespan) { setTime(st.makespan); st.playing = false; updateButtons(); emit('end', tickInfo()); }
        else setTime(nt);
      }
      draw(ms); updateReadout(); emit('tick', tickInfo());
      if (st.playing) st.raf = requestAnimationFrame(frame);
    }
    function schedule() { if (!st.raf && !destroyed) st.raf = requestAnimationFrame(frame); }

    var api = {
      play: function () {
        if (st.playing) return api;
        if (st.tau >= st.makespan) setTime(0);
        st.playing = true; st.lastMs = 0; updateButtons(); emit('play', tickInfo()); schedule(); return api;
      },
      pause: function () { st.playing = false; updateButtons(); emit('pause', tickInfo()); schedule(); return api; },
      toggle: function () { return st.playing ? api.pause() : api.play(); },
      /** Advance (paused) to the next event start after the current time. */
      step: function () {
        st.playing = false;
        var i = upperBound(st.ev, st.tau);
        setTime(i < st.n ? st.ev[i].t : st.makespan);
        updateButtons(); updateReadout(); schedule(); return api;
      },
      restart: function () { setTime(0); st.lastMs = 0; updateButtons(); updateReadout(); schedule(); return api; },
      setSpeed: function (nsPerSecond) {
        st.speed = Math.max(1, Math.min(1e9, num(nsPerSecond, st.speed))); st.speedExplicit = true;
        speedIn.value = String(Math.log10(st.speed)); updateReadout(); schedule(); return api;
      },
      getSpeed: function () { return st.speed; },
      /** seek(f) with 0<=f<=1 is a fraction of the makespan; larger values (or {t: ns}) are ns. */
      seek: function (v) {
        var t;
        if (v && typeof v === 'object') t = v.t != null ? v.t : (v.frac || 0) * st.makespan;
        else { v = num(v, 0); t = v >= 0 && v <= 1 ? v * st.makespan : v; }
        setTime(t); st.lastMs = 0; updateButtons(); updateReadout(); schedule(); return api;
      },
      load: function (events) { load(events); return api; },
      totals: function () { return simulate(st.ev, C, layout); },
      time: function () { return st.tau; },
      state: tickInfo,
      on: function (name, fn) {
        (listeners[name] || (listeners[name] = [])).push(fn);
        return function () { var ls = listeners[name]; var i = ls.indexOf(fn); if (i >= 0) ls.splice(i, 1); };
      },
      /** Draw the current time synchronously (frames normally run via requestAnimationFrame). */
      render: function () { draw(typeof performance !== 'undefined' ? performance.now() : 0); updateReadout(); return api; },
      refreshTheme: function () { theme = readTheme(); buildLegend(); buildStatic(); schedule(); return api; },
      resize: function () { fit(); schedule(); return api; },
      destroy: function () {
        destroyed = true; st.playing = false;
        if (st.raf) cancelAnimationFrame(st.raf);
        if (ro) ro.disconnect(); else window.removeEventListener('resize', onResize);
        if (mq) { if (mq.removeEventListener) mq.removeEventListener('change', onTheme); else mq.removeListener(onTheme); }
        if (rootEl.parentNode) rootEl.parentNode.removeChild(rootEl);
      },
      element: rootEl, canvas: canvas, constants: C
    };

    bPlay.addEventListener('click', function () { api.toggle(); });
    bStep.addEventListener('click', function () { api.step(); });
    bRestart.addEventListener('click', function () { api.restart(); });
    speedIn.addEventListener('input', function () {
      st.speed = Math.pow(10, +speedIn.value); st.speedExplicit = true; updateReadout(); schedule();
    });
    scrub.addEventListener('input', function () {
      st.scrubbing = true; setTime(+scrub.value / 1000 * st.makespan); st.lastMs = 0; updateButtons(); updateReadout(); schedule();
    });
    scrub.addEventListener('change', function () { st.scrubbing = false; });
    var onResize = function () { var w = container.clientWidth; if (w && w !== W) { fit(); schedule(); } };
    var ro = typeof ResizeObserver === 'function' ? new ResizeObserver(onResize) : null;
    if (ro) ro.observe(container); else window.addEventListener('resize', onResize);
    var mq = typeof matchMedia === 'function' ? matchMedia('(prefers-color-scheme: dark)') : null;
    var onTheme = function () { api.refreshTheme(); };
    if (mq) { if (mq.addEventListener) mq.addEventListener('change', onTheme); else mq.addListener(onTheme); }

    fit();
    if (config.speed != null) st.speed = Math.max(1, Math.min(1e9, num(config.speed, 1000)));
    load(config.events || []);
    if (config.autoplay !== false && !reduced && st.n > 0) api.play();
    return api;
  }

  return { create: create, simulate: simulate, normalize: normalize, manhattan: manhattan,
           fmt: fmt, DEFAULT_CONSTANTS: DEFAULT_CONSTANTS, VERSION: VERSION };
}));
