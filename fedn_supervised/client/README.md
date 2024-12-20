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

## Overwrite config

To change the [hydra config](https://hydra.cc/docs/intro/), add all arguments to ```client/overrides.txt``` without any empty line. 
 
 ### Example

overrides.txt
 ```
 model.dropout=0.1
 model.pretrained_checkpoint=<insert path here>
 ```

 # Create FEDn packages

 ```
 uv run fedn package create --path client
 ```

# Run 1-6 clients clients


Download 6 client.yaml files [here](https://fedn.scaleoutsystems.com/marfr65/plantmap-olf/clientlist) and store them as ```client_files/cliend{1..6}.yaml```.
On berzelius:
```
chmod +x run.sh
./run.sh
```

Alternatively, you can run ```sbatch exp.sh <client ID>```.
```
chmod +x exp.sh
./exp.sh i
```
where ```i``` is a number between 1 to 6. 