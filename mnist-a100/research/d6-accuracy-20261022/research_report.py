"""Generate the D6 report from retained audits; never selects or executes learners."""
import json
from pathlib import Path


def number(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def interval(value):
    if not value:
        return 'pending'
    low, high = value['approximate_95pct_t_interval']
    return f'[{number(low)}, {number(high)}]'


def render(root, final=False):
    root=Path(root)
    analysis=json.loads((root/'analysis.json').read_text())
    confirm_path=root/'confirmation-analysis.json'
    confirmation=json.loads(confirm_path.read_text()) if confirm_path.exists() else None
    plan_path=root/'confirmation-plan.json'
    confirm_plan=json.loads(plan_path.read_text()) if plan_path.exists() else None
    selected=analysis.get('selected')
    oversized=[arm for arm in (confirm_plan or {}).get('arms',[]) if arm.get('source_bytes',0)>20480]
    size_note=' '.join(f"The research source for {', '.join(arm.get('roles',[arm['name']]))} is {arm['source_bytes']:,} bytes, exceeding the official 20,480-byte source cap." for arm in oversized)
    lines=['# D6 accuracy research 22 October 2026','']
    if final:
        assert analysis.get('all_development_finalized') is True and selected
        assert confirmation and confirmation.get('all_shards_finalized') is True
        assert confirmation.get('all_arms_valid') is True and not confirmation.get('global_errors')
        rolemap=confirmation['role_to_algorithm']
        chosen=next(arm for arm in confirmation['arms'] if arm['name']==rolemap['selected'])
        assert chosen['valid_confirmation'] and chosen['observed_total']==110000
        reached='met' if chosen['target_met'] else 'did not reach'
        lines += [f"The frozen `{confirm_plan['selected_development_arm']}` ensemble ({confirm_plan['selected_prefix']} members) achieved **{number(chosen['observed_accuracy_pct'])}%** ({chosen['pooled_correct']}/110000) on eleven fresh released MNIST draws; the approximate 95% draw-level t interval is **{interval(chosen['accuracy_draw_interval'])}%**. Mean / maximum measured algorithm time was **{number(chosen['mean_cumulative_algorithm_seconds'])} / {number(chosen['max_cumulative_algorithm_seconds'])} seconds per draw**. It {reached} the exploratory 98.6% target within the frozen research budget. This is an achieved operating point among tested methods, not an established accuracy ceiling.", '']
        for contrast in confirmation['comparisons']:
            if contrast['left_role']=='selected' and contrast['right_role']=='baseline450':
                continue
            estimate=contrast['paired_accuracy_difference_pp_interval']
            assert estimate is not None
            low,high=estimate['approximate_95pct_t_interval']
            interpretation=('supports a gain for this specific regularizer' if low>0 else
                            'supports a loss for this specific regularizer' if high<0 else
                            'does not clearly establish a gain or loss for this specific regularizer')
            role=contrast['left_role']
            scope=('The selected learner is the regularized comparison.' if role=='selected' else
                   'The overall development winner is unregularized; a separately frozen best-regularized secondary candidate supplies this comparison.')
            lines += [f"**Regularization effect:** `{role}` minus `{contrast['right_role']}` was **{number(estimate['mean'])} percentage points**, with approximate 95% paired draw-level interval **{interval(estimate)}**. This {interpretation}. {scope} The matched control preserves training duration, width, guard and ensemble size, removing only explicit matrix L2, hidden dropout, hidden-encoder DropConnect or extra Gaussian noise as recorded in the frozen plan.", '']
        interpretation_path=root/'maintainer-interpretation.json'
        if interpretation_path.exists():
            interpretation=json.loads(interpretation_path.read_text())
            if interpretation.get('d6_target_recommendation'):
                lines += [interpretation['d6_target_recommendation'],'']
    else:
        lines += ['**Provisional:** final accuracy and regularization conclusions await all nineteen terminal development outcomes and the frozen eleven-draw fresh confirmation. No final best achievable accuracy is inferred from the development ranking.', '']
    lines += ['## Fresh confirmation','']
    if confirmation:
        lines += ['| Frozen role(s) | Correct / total | Accuracy | Approximate95% draw interval | Mean / maximum algorithm seconds | Audit |',
                  '|---|---:|---:|---:|---:|---|']
        for arm in confirmation['arms']:
            lines.append(f"| {', '.join(arm['roles'])} | {arm['pooled_correct']}/{arm['observed_total']} | {number(arm['observed_accuracy_pct'])}% | {interval(arm.get('accuracy_draw_interval'))} | {number(arm.get('mean_cumulative_algorithm_seconds'))} / {number(arm.get('max_cumulative_algorithm_seconds'))} | {'valid' if arm.get('valid_confirmation') else 'pending/invalid'} |")
        lines += ['', 'Intervals use eleven paired draw-level observations, conditional on this fixed MNIST pool and frozen procedure. Overlapping query images do not constitute 110000 independent Bernoulli trials. The source, prefix, roles and draw seeds were frozen before the fresh inputs were prepared; no confirmation-based reselection is allowed.']
    else:
        lines += ['Pending. The frozen protocol compares the selected learner with the 450-step baseline. It also compares a regularized candidate with its matched unregularized control; if the overall winner has no extra regularizer, that candidate is a predeclared secondary choice.']
    lines += ['', '## Development screen','',
              f"Terminal outcomes retained: **{analysis.get('finalized_logical_arms',0)}/19**. Initial ten arms used three development draws. The adaptive nine-arm no-guard extension reuses those seeds, with 7200 steps, sixteen sequential members and prefixes 1/2/4/8/16; all plan amendments are retained. A candidate must stay within 300 seconds on every development draw. The ranking selects pooled correct count, then mean reported call time; selected development accuracy is optimistic and is kept separate from fresh confirmation.", '',
              '| Arm | Phase | Best eligible prefix | Correct / total | Accuracy | Mean / maximum budget seconds |',
              '|---|---|---:|---:|---:|---:|']
    candidates=analysis.get('all_prefix_candidates',[])
    for arm in analysis.get('arms',[]):
        eligible=[c for c in candidates if c['arm']==arm['name'] and c.get('eligible')]
        best=min(eligible,key=lambda c:(-c['pooled_correct'],c['mean_full_call_seconds'],c['members'])) if eligible else None
        if best:
            lines.append(f"| {arm['name']} | {arm['phase']} | {best['members']} | {best['pooled_correct']}/{best['total']} | {number(best['accuracy_pct'])}% | {number(best.get('mean_budget_seconds'))} / {number(best.get('max_budget_seconds'))} |")
        else:
            lines.append(f"| {arm['name']} | {arm['phase']} | — | {'terminal; no eligible prefix' if arm.get('finalized') else 'pending'} | — | — |")
    lines += ['', '## Method and interpretation','',
              'The variants derive from spartanhaden’s PR #139 kernel-taught Ladder: its learned teacher supplies transformed features and soft targets to the student, followed by the inherited hidden-space HKNN correction. The study varies training duration, probability ensembles, matrix L2, hidden dropout, hidden-encoder DropConnect, Gaussian noise and first-hidden-layer width. Each member is fitted from the supplied draw. The learner receives only train inputs, train labels and query inputs; the outer controller retains query truth.', '',
              'The original guard can substitute teacher probabilities when training errors exceed 40 or class disagreements cross its threshold. Stronger regularization often activates it, so the extension removes the final substitution while retaining teacher features, soft targets and HKNN. Matrix L2 adds `(lambda/2)*sum(weight**2)` to encoder/decoder matrices only. Dropout and DropConnect operate only during training. These comparisons measure the specified interventions rather than all possible regularization.', '',
              'Initial `dropout7200` and `long28800` have released-coordinate hashes different from the other initial arms, with matching query labels. They remain retained, and those initial comparisons are not claimed to be byte-exact paired inputs. Extension and confirmation instead share frozen per-seed NPZ arrays, checked before use. Initial timing counts member `_predict` calls; future `cumulative_algorithm_seconds` also includes synchronized probability accumulation and argmax. Both exclude draw preparation, warmup, truth comparisons and evidence transfer/serialization.', '',
              f'The five-minute research budget differs from the stock scorer’s 60-second per-call watchdog. This study changes no competition rules and supplies no official D6 score, energy, sandbox warmup or foreign-data qualification. Source files use the research `_predict(...) -> (probabilities, training_errors)` interface and are not established as label-only submission entrypoints. {size_note} Public evidence consists of marked excerpts with labels, counts, timing and hashes; omitted probability payloads and input NPZs remain in the complete local archive. The public verifier replays excerpt counts and integrity; retained local audits additionally check probability averaging and array bytes. Controller chronology and infrastructure limitations are recorded in provenance.', '',
              '## Sources and evidence','',
              '- Algorithm: [PR #139](https://github.com/cybertronai/sutro-problems/pull/139), preserving its direct and inherited credits; contributor list is retained in provenance.',
              '- Discussion: [longer training and ensembles](https://t.me/sutro_group/2993/3186), [regularization request](https://t.me/sutro_group/2993/3198). These discussion claims are motivation, not independent measurements in this report.',
              '- [Plans](plan.json), [extension amendments](extension-plan.json), [development audit](analysis.json), [provenance](provenance.json).',
              '- [Public evidence inventory](evidence-manifest.json), [publication policy](publication-policy.json), [public verifier](verify_public_evidence.py). Run `python verify_public_evidence.py` in this package.']
    if confirm_plan:
        lines.append('- [Frozen confirmation plan](confirmation-plan.json), [fresh-confirmation audit](confirmation-analysis.json).')
    for name in ['confirmation-results.png','regularization-effect.png']:
        if (root/name).exists(): lines += ['',f'![Fresh-confirmation results]({name})']
    if (root/'development-frontier.png').exists():
        lines += ['', '![Development accuracy and budget](development-frontier.png)']
    lines += ['', '[Competition review: four merged PRs and seven held submissions](../../reviews/20261022/README.md)']
    return '\n'.join(lines)+'\n'
