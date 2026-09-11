"""Export visible conversation messages only; omit internal prompts and reasoning.

Usage: python export_session.py --input /path/to/rollout.jsonl --output DIR
The source JSONL is deliberately not copied or published.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    messages = []
    started = False
    session_id = None
    for raw in a.input.read_text().splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue  # A live log can end with an incomplete line.
        data = row.get('payload', {})
        if row.get('type') == 'session_meta':
            session_id = data.get('session_id') or data.get('id')
        if row.get('type') != 'response_item' or data.get('type') != 'message':
            continue
        role = data.get('role')
        value = '\n'.join(x.get('text', '') for x in data.get('content', [])
                          if x.get('type') in ('input_text', 'output_text', 'text'))
        if role == 'user' and value.startswith('Attempt a submission'):
            started = True
        if not started:
            continue
        if role == 'assistant':
            # Positive allowlist: no hidden reasoning or ambiguous message channels.
            if data.get('phase') not in ('commentary', 'final_answer', 'final'):
                continue
        elif role == 'user':
            if any(tag in value for tag in ('<environment_context>', '<recommended_plugins>',
                                           '<subagent_notification>', '<turn_aborted>')):
                continue
        else:
            continue
        messages.append({'timestamp_utc': row.get('timestamp'), 'role': role, 'text': value})
    if not messages:
        raise ValueError('No matching user-visible conversation was found')
    cutoff = datetime.now(timezone.utc).isoformat()
    record = {'session_id': session_id, 'exported_at_utc': cutoff,
              'scope': 'Visible user/assistant messages from the submission request through the export cutoff. '
                       'Internal reasoning, system/developer instructions, environment metadata and raw tool payloads are excluded. '
                       'Measured results and selected commands are preserved separately in the report and source artifacts.',
              'messages': messages}
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'session.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    text = ['# MNIST-small submission session', '',
            f'Exported at **{cutoff}**. Session `{session_id}`.', '', record['scope'], '',
            'This is a conversation export, not a reconstruction of private reasoning. '
            'It is a snapshot: publication checks after this cutoff are recorded in the accompanying report.', '']
    for n, message in enumerate(messages, 1):
        text += [f'## {n}. {message["role"].title()} · {message["timestamp_utc"]}', '', message['text'], '']
    worklog = a.output / 'worklog.md'
    if worklog.exists():
        text += ['', worklog.read_text()]
    (a.output / 'session.md').write_text('\n'.join(text), encoding='utf-8')
    print(f'Exported {len(messages)} visible messages; internal logs were not copied.')


if __name__ == '__main__':
    main()
