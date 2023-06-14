# Medical-Insurance-Workflow-Orchestration-using-Prefect


## Machine-Learning-Workflow-Orchestration-using-Prefect
### Project Explanation

This is a simple linear regression project to predict medical insurance costs based on several factors that include age, sex, BMI, region, smoker status and number of children.

The project structure is as follows:
```
MEDICALINSURANCE
├── main.py
├── app.py
├── templates
│   └── index.html
└── static
    ├── css
    │   └── styles.css
    └── js
        └── script.js
```
The project structure consists of several files and folders. The main code and workflow orchestration are implemented in the main.py file. This file contains the core logic for the model and coordinates the execution of the project.

The app.py file is responsible for handling the Flask application. It contains the necessary code to set up the server, define routes, and handle incoming requests from the web app.

The templates folder houses the index.html file. This HTML file serves as the main page for the web app. It contains a form where users can submit their inputs, such as age, sex, BMI, region, smoker status, and the number of children. These inputs are then utilized by the model to calculate insurance costs.

The static folder is divided into two subfolders: css and js. The css folder contains the styles.css file, which holds the CSS code responsible for styling the web page. It determines the visual appearance of the form and other elements on the page. On the other hand, the js folder contains the script.js file, which includes JavaScript code for handling event listeners related to the form in the index.html file. This JavaScript code enables interactivity and responsiveness of the web app.

When combined, these files and folders create the structure of the project, allowing for the seamless integration of the model and the web app components.
