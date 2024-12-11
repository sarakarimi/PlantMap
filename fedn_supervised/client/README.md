# Getting started

## Install uv

uv is a replacement of conda and pip.
No further configuration is necessary, once ```uv run <module>``` is called the first time, the environment will be set-up. 

```
# on linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Set number of agents and agent ID of the current agent.

```
export NUM_AGENTS=< insert number here >
export AGENT AGENT_ID=< insert agent number here less than NUM_AGENTS - 1 >
```

## Run FEDn steps individually

```
# create initial, pretrained model
uv run fedn run build --path client

# train a single epoch (model is always saved as seed.npz)
uv run fedn run train --path client --input seed.npz --output trained.npz

# validate on train and validation data
uv run fedn run train --path client --input trained.npz --output output.json

```

