## Flux Model 

#### Owner Tim Whittaker => timothy.whittaker@datarobot.com

Install `Flux` and `BSON` in your Julia environment if not already available

Again, this is a slow startup time.  For scoring and serving it is fine, but it will always take some time to start everything up, therefore the recommendation is to use a system image of your environment to speed up things considerably.  

See details [here](https://github.com/datarobot/datarobot-user-models/tree/master/public_dropin_environments/julia_mlj) on how that can be accomplished.  

## Scoring

`drum score --code-dir ./boston-flux-nn --target-type regression --input data/boston_housing.csv --verbose --logging-level info`

## Serving with Docker

`drum server --code-dir ./boston-flux-nn --target-type regression --verbose --logging-level info`






