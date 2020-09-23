# MLOps Agent - Python End to End

This notebook provides an example of how you can use the MLOps Agents to monitor external deployments using DataRobot.

### Getting Started
Open `MLOps Agent - Python End to End.ipynb` and follow isntructions. Alternatively, you can execute this script using [Google colab](https://colab.research.google.com/drive/1fS_sy2L96PsHHpSJRGst2ZzNILyIWAbb?usp=sharing#scrollTo=4MHiEsQX8dTP)

### Requirements

In DataRobot, navigate to your deployment of the external model:
- Click on the Deployment itself.
- Then click on "Integrations and follow the instructions you see for your selected language (Python, Java, R).
- You will need to install the MLOPs Library. In python this is done with: 
  - pip install <unpacked DataRobot's Monitoring Agent tar file>/lib/datarobot_mlops-*-py2.py3-none-any.whl


Usage:
    DataRobot's Monitoring Agent is an advanced feature that enables monitoring for models deployed
    outside of DataRobot. Follow these steps to use this feature.
    1. Download and extract DataRobot's Monitoring Agent tar file. This is available through the
       DataRobot MLOps UI via User icon -> "Developer Tools" -> "External Monitoring Agent".
    2. Open your preferred browser. In toolbar, click "File" -> "Open File".
       Then choose this file <your unpacked directory>/docs/html/quickstart.html.
    3. Follow the "Quick Start" instructions to set up Monitoring Agent.
    4. This example uses the DataRobot's MLOps library which you can install with:
       pip install <unpacked DataRobot's Monitoring Agent tar file>/lib/datarobot_mlops-*-py2.py3-none-any.whl
    5. MLOps library requires the following parameters to be provided:
       deployment ID, model ID, output type [STDOUT, OUTPUT_DIR, NULL].
       If output type is OUTPUT_DIR, which means that Monitoring Agent will send data to the DataRobot MLOPs service,
       the following parameters also have to be configured: spool directory, max size per spool file,
       max number of spool files. Spool directory path must match the Monitoring Agent path configured by admin.
       For advanced usage, see the examples.
       These parameters can be configured with APIs, check the MLOps library init() call,
       or with environment variables. For example:
       # export MLOPS_DEPLOYMENT_ID='YOUR_DEPLOYMENT_ID'
       # export MLOPS_MODEL_ID='YOUR_DEPLOYMENT_ID'
       # export MLOPS_OUTPUT_TYPE=OUTPUT_DIR
       # export MLOPS_SPOOLER_DIR_PATH=/tmp/ta
       # export MLOPS_SPOOLER_FILE_MAX_SIZE=104857600
       # export MLOPS_SPOOLER_MAX_FILES=5

       Notes:
              - parameter configuration via the API takes precedence over environment variables.
              - for testing purposes, you can start with STDOUT output type,
                which doesn't require providing any of the spooler-related parameters.
    6. Run current snippet:
       python datarobot-report-stats.py
