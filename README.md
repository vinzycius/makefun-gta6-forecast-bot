# makefun-gta6-forecast-bot

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/vinzycius/makefun-gta6-forecast-bot/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/vinzycius/makefun-gta6-forecast-bot/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day (2026-11-19 UTC), hours 2–24, as five quantiles each. Python 3.13, standard library only, no network. It reads one case file (`/input/<id>.in`) and writes one forecast (`/output/<id>.out`).

The model is fit at run time from the case's own comparables (past big game launches' hourly Twitch curves): per hour it regresses the log ratio to hour 1 on the log hour-1 print — bigger launches peak lower relative to hour 1 — blends that with the comparables' median shape, and widens the quantile spread with the horizon. On the four practice cases it scores a mean pinball loss of 0.139, against the house persistence baseline's 0.387.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=gta6-twitch-ccu-forecast-01-kit/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
