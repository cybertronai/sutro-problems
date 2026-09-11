"""Export only visible conversation and structured user clarification answers.

The input JSONL is read locally and never copied. Assistant analysis, tool
payloads, internal prompts and environment/AGENTS metadata are not exported.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

VISIBLE_ASSISTANT_CHANNELS = {'commentary', 'final', 'final_answer'}
TEXT_TYPES = {'input_text', 'output_text', 'text'}
START_PREFIX = 'Make a Submission for the medium'
METADATA_MARKERS = (
    '<environment_context>', '<recommended_plugins>', '<subagent_notification>',
    '<turn_aborted>', '<app-context>', '<skills_instructions>',
    '<permissions instructions>', '<collaboration_mode>', '<multi_agent_role>',
    '<multi_agent_mode>', '<INSTRUCTIONS>', '# AGENTS.md instructions',
)
REPLY = re.compile(r'^\s*<send_user_message_question_reply>\s*(.*?)\s*'
                   r'</send_user_message_question_reply>\s*$', re.DOTALL)


def utc(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('The cutoff must include a timezone')
    return parsed.astimezone(timezone.utc)


def visible_text(payload):
    """Positive role/channel/content allowlist, followed by metadata rejection."""
    if payload.get('type') != 'message':
        return None
    role = payload.get('role')
    if role == 'assistant':
        channel = payload.get('phase', payload.get('channel'))
        if channel not in VISIBLE_ASSISTANT_CHANNELS:
            return None
    elif role != 'user':
        return None
    parts = []
    for block in payload.get('content', []):
        if block.get('type') not in TEXT_TYPES or not isinstance(block.get('text'), str):
            continue
        value = block['text'].strip()
        if not value or any(marker in value for marker in METADATA_MARKERS):
            continue
        if role == 'user' and REPLY.fullmatch(value):
            answers = json.loads(REPLY.fullmatch(value).group(1))
            if not isinstance(answers, list):
                raise ValueError('Unexpected structured clarification format')
            lines = []
            for answer in answers:
                question, reply = answer.get('question'), answer.get('answer')
                if not isinstance(question, str) or not isinstance(reply, str):
                    raise ValueError('Structured clarification lacks visible question/answer text')
                # IDs and transport metadata intentionally have no output path.
                lines.extend([f'**Question:** {question}', '', f'**Answer:** {reply}'])
            parts.append(('clarification', '\n'.join(lines)))
        elif role == 'user' and value.startswith('<'):
            # Other synthetic wrappers are not positively identified user prose.
            continue
        else:
            parts.append(('message', value))
    if not parts:
        return None
    return {'role': role, 'kind': 'clarification' if all(p[0] == 'clarification' for p in parts) else 'message',
            'text': '\n\n'.join(p[1] for p in parts)}


def extract(source, cutoff, start_prefix=START_PREFIX):
    messages = []
    started = False
    session_id = None
    for raw in source.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue  # A live rollout may end with a partially written line.
        payload = row.get('payload', {})
        if row.get('type') == 'session_meta':
            session_id = payload.get('session_id') or payload.get('id')
        # Event records can duplicate messages; tools and events are excluded.
        if row.get('type') != 'response_item' or not row.get('timestamp'):
            continue
        if utc(row['timestamp']) > cutoff:
            continue
        message = visible_text(payload)
        if message is None:
            continue
        if message['role'] == 'user' and message['text'].casefold().startswith(start_prefix.casefold()):
            started = True
        if started:
            messages.append({'timestamp_utc': row['timestamp'], **message})
    if not started or not messages:
        raise ValueError('No matching visible medium-submission request was found before the cutoff')
    return session_id, messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--cutoff', help='Inclusive ISO8601 timestamp; defaults to current UTC time')
    parser.add_argument('--start-prefix', default=START_PREFIX)
    args = parser.parse_args()
    cutoff = utc(args.cutoff) if args.cutoff else datetime.now(timezone.utc)
    session_id, messages = extract(args.input, cutoff, args.start_prefix)
    scope = ('Visible user and assistant messages from the medium-submission request through the stated cutoff, '
             'including user answers to asynchronous clarification questions. Internal reasoning, system/developer '
             'instructions, AGENTS/environment metadata, raw tools and the source session file are excluded. '
             'The sequence preserves target changes; the accompanying report states the final evaluated task.')
    record = {'session_id': session_id, 'exported_at_utc': datetime.now(timezone.utc).isoformat(),
              'cutoff_utc': cutoff.isoformat(), 'start_prefix': args.start_prefix,
              'scope': scope, 'messages': messages}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'session.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    content = ['# MNIST-medium submission session', '', f'Visible conversation through **{cutoff.isoformat()}**.',
               '', scope, '', 'This snapshot contains visible conversation, not private reasoning. '
               'Measurements and reproduction instructions are preserved in the accompanying report.', '']
    for index, message in enumerate(messages, 1):
        label = 'User clarification' if message['kind'] == 'clarification' else message['role'].title()
        content.extend([f'## {index}. {label} · {message["timestamp_utc"]}', '', message['text'], ''])
    (args.output / 'session.md').write_text('\n'.join(content), encoding='utf-8')
    print(json.dumps({'visible_messages': len(messages),
                      'clarification_answers': sum(m['kind'] == 'clarification' for m in messages),
                      'cutoff_utc': cutoff.isoformat(), 'raw_session_copied': False}, indent=2))


if __name__ == '__main__':
    main()
