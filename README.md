# MLOps Examples

This repository contains multiple examples concerning DataRobot's Machine Learning Operations (MLOps) solution. The scripts can be followed end to end and customised depending on the problem you are trying to solve.

## Usage

For each respective guide, follow the instructions in its own `.ipynb` or `.py` file. There will also be a `requirements.txt` file in each folder with instructions on how to create an environment to run everything successfully.

Here is some explanation of the different definitions used throughout: 
- **MLOps Tracking Agents**: MLOps Tracking Agents are used when you want to deploy external models and monitor them in DataRobot. For example, you have a custom (or DataRobot) model and you deploy it in your own Kubernetes cluster (or anywhere really). In those cases, MLOps tracking agents will sent statistics back to DataRobot so that you can still monitor your model's accuracy, service health, data drift, etc.
- **MLOps DRUM overview**: MLOps DRUM is an open-source framework created and managed by DataRobot that allows you to easily deploy custom models. It provides out of the box consistency & validity checks, as well as single command deployment. DRUM is also seamlessly integreated with the DataRobot platform. If you use the framework, then you can use your custom models directly within the DataRobot platform. Here is the official Github repository for [DRUM](https://github.com/datarobot/datarobot-user-models).
- **Custom Inference Models**: End to end examples of custom modeling code and how it is structured in order to be deployable using the DataRobot platform. The custom code here is basically taking advantage of the DRUM framework mentioned above.
- **Custom Tasks**: With Composable AI, DataRobot allows you to manipulate DataRobot created blueprints and add your own custom preprocessing steps. Within custom tasks, there are examples of how your code needs to look like to achieve this.

Some of the notebooks can also be executed through Google Colab.

## Important Links

- To learn to use DataRobot, visit [DataRobot University](https://university.datarobot.com/)
- For General articles on DataRobot and news, visit [DataRobot Community](https://community.datarobot.com/)
- End to end DataRobot API examples [Tutorials for Data Scientists](https://github.com/datarobot-community/tutorials-for-data-scientists)
- DataRobot API examples [Examples for Data Scientists](https://github.com/datarobot-community/examples-for-data-scientists)

## Contents

### MLOps Tracking Agents Overview
- *MLOps Tracking Agent Notebook*: An example of how you can use DataRobot's MLOps Agents functionality to monitor external deployments. [Python](https://github.com/datarobot-community/custom-models/tree/master/tracking_agents/python)

### MLOps DRUM Overview
- *MLOps DRUM Notebook*: An example of you can use the DataRobot Model Runner (DRUM) library to test your custom models before deploying them using DataRobot. [Python](https://github.com/datarobot-community/mlops-examples/tree/master/MLOps%20DRUM)

### Custom Inference Model Examples
- *Custom Inference Models*: Examples in multiple languages on how to create custom inference models. Some of the scripts have been updated to also include the code needed to run this as a custom training model: [Multiple Languages](https://github.com/datarobot-community/mlops-examples/tree/master/Custom%20Model%20Examples/Readmissions)

### Custom Tasks
- *Custom Tasks*: Examples of custom-tasks that you can use directly within the DataRobot platform to manipulate blueprints. Check out how they look like and create your own tasks! (Multiple Languages)(https://github.com/datarobot-community/custom-models/tree/master/custom_tasks)


## Setup/Installation

Each project folder contains its own instructions on setup and requirements. Furthermore, instructions are also conveniently added to the scripts themselves so that users do not need to share the readme file.

## Development and Contributing

If you'd like to report an issue or bug, suggest improvements, or contribute code to this project, please refer to [CONTRIBUTING.md](CONTRIBUTING.md).


# Code of Conduct

This project has adopted the Contributor Covenant for its Code of Conduct. 
See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) to read it in full.

# License

Licensed under the Apache License 2.0. 
See [LICENSE](LICENSE) to read it in full.


