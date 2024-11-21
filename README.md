# DL Energy Experiment CLI

## Usage

```bash
usage: experiment.py [-h] [-e EXPERIMENT] [--task {vision}] [--source SOURCE] [--type TYPE]
                     [--family FAMILY][--model MODEL] [-r REPETITIONS] [--id ID] [-p] [-f FORCE]

optional arguments:
  -h, --help            show this help message and exit
  -e EXPERIMENT, --experiment EXPERIMENT
                        Name the Experiment
  --type TYPE           Test all models of the same kind
  --family FAMILY       Test all models of the same family
  --model MODEL         Test a specific model
  -r REPETITIONS, --repetitions REPETITIONS
                        Experiment repetitions
  --id ID               GPU ID
  -p, --plot_only       Plot results only
  -f FORCE, --force FORCE
                        Force the node name for plotting data not run on the same node


```
