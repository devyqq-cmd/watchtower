#!/bin/zsh
# 对比 MiniMax vs Claude CLI 分析质量
# 在普通终端运行（不要在 Claude Code 里运行）：
#   zsh scripts/compare_ai.sh

set -a && source .env && set +a

~/.local/bin/uv run --project . python - << 'PYEOF'
from ai.analyst import AINarrativeAnalyst, PROMPT_TEMPLATE
import subprocess, os

data = {
    'risk_score': 52, 'rsi': 25.7, 'z_dist': -2.42, 'price': 380.0,
    'rule_id': 'OVERSOLD_OPP', 'severity': 'buy'
}
prompt = PROMPT_TEMPLATE.format(
    symbol='0700.HK', rule_id=data['rule_id'], severity=data['severity'],
    risk_score=data['risk_score'], rsi=data['rsi'], z_dist=data['z_dist'], price=data['price']
)

print("=" * 60)
print("MiniMax (MiniMax-Text-01)")
print("=" * 60)
a = AINarrativeAnalyst()
print(a._call_minimax(prompt, '0700.HK', 52, 25.7, -2.42))

print()
print("=" * 60)
print("Claude CLI (claude -p)")
print("=" * 60)
env = os.environ.copy()
env.pop('CLAUDECODE', None)
r = subprocess.run(['claude', '-p', prompt], capture_output=True, text=True, timeout=60, env=env)
if r.returncode == 0:
    print(r.stdout.strip())
else:
    print(f"Error: {r.stderr.strip()}")
PYEOF
