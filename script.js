// static/js/script.js

// Function to handle form submission
function handleSubmit(event) {
    event.preventDefault();
  
    // Retrieve form inputs
    const age = document.getElementById("age").value;
    const sex = document.getElementById("sex").value;
    const bmi = document.getElementById("bmi").value;
    const children = document.getElementById("children").value;
    const smoker = document.getElementById("smoker").value;
    const region = document.getElementById("region").value;
  
    // Perform any additional validation if needed
  
    // Send form data to the server
    fetch("/prediction", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        age: age,
        sex: sex,
        bmi: bmi,
        children: children,
        smoker: smoker,
        region: region,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        // Display the prediction result
        const resultElement = document.getElementById("result");
        resultElement.innerHTML = `The insurance cost is USD ${data.prediction}`;
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  }
  
  // Attach event listener to the form submit button
  const form = document.getElementById("insurance-form");
  form.addEventListener("submit", handleSubmit);
  