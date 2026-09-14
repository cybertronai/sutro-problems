const menu = document.querySelector('#contents-menu');
const mobile = window.matchMedia('(max-width: 760px)');
function syncMenu() { menu.open = !mobile.matches; }
syncMenu();
mobile.addEventListener('change', syncMenu);
menu.addEventListener('click', (event) => {
  if (event.target.closest('a') && mobile.matches) menu.open = false;
});
const links = [...document.querySelectorAll('.contents nav a[href^="#section-"], .contents nav a[href="#appendix"]')];
const headings = links.map(link => document.querySelector(link.getAttribute('href')));
const progress = document.querySelector('.reading-progress span');
let scheduled = false;
function updateReadingPosition() {
  const height = document.documentElement.scrollHeight - window.innerHeight;
  progress.style.width = `${height > 0 ? Math.min(100, Math.max(0, window.scrollY / height * 100)) : 0}%`;
  let current = -1;
  headings.forEach((heading, index) => { if (heading.getBoundingClientRect().top <= 150) current = index; });
  links.forEach((link, index) => {
    if (index === current) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
  scheduled = false;
}
window.addEventListener('scroll', () => {
  if (!scheduled) { scheduled = true; requestAnimationFrame(updateReadingPosition); }
}, { passive: true });
function updateTables() {
  document.querySelectorAll('.table-scroll, .equation').forEach(table => {
    const overflowing = table.scrollWidth > table.clientWidth + 1;
    table.previousElementSibling.hidden = !overflowing;
    table.tabIndex = overflowing ? 0 : -1;
  });
  updateReadingPosition();
}
window.addEventListener('resize', updateTables);
document.fonts.ready.then(updateTables);
updateTables();
