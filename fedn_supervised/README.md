# Finetuning Flower Classification using FEDn

## Preparation

```
mkdir client_files
```
Go to the FEDn [website](https://fedn.scaleoutsystems.com/projects/), navigate to the clients page in your project and download six client yaml files. 
Save these files under ```client_files/client{0..5}.yaml```.

## Create the initial seed file
Run ```uv run fedn run build --path client``` and you will get a ```client/seed.npz```.
In your FEDn project, start a new session and upload the seed file. 

## Start all six clients
In bash, you need to define the number of clients and the client ID for each client.
```
export NUM_AGENTS=6
export AGENT_ID={0..5}
```
After that, run 
```
uv run --cache-dir uv_cache/ fedn client start -in client_files/client{0..5}.yaml --secure=True --force-ssl --local-package
```
This will start the client. 
If you want to run everything on slurm devices, take a look at ```exp.sh``` example. 

## Start Training

You should be able to see all clients connecting to the host. 
If that is the case, you can navigate inside the session tab inside FEDn to start the training process. 