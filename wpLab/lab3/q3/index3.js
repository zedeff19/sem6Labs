document.getElementById('marksForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form from submitting and page reloading
  
    // Access the input values
    const subject1 = parseFloat(document.getElementById('subject1').value);
    const subject2 = parseFloat(document.getElementById('subject2').value);
    const subject3 = parseFloat(document.getElementById('subject3').value);
    const subject4 = parseFloat(document.getElementById('subject4').value);
  
    // Check if the values are valid numbers
    if (isNaN(subject1) || isNaN(subject2) || isNaN(subject3) || isNaN(subject4)) {
      alert('Please enter valid marks for all subjects');
      return;
    }
  
    // Calculate total and average
    const totalMarks = subject1 + subject2 + subject3 + subject4;
    const averageMarks = totalMarks / 4;
  
    let grade = '';
    if (averageMarks > 90)
    {
        grade = 'A'
    }
    else if(averageMarks > 80)
        grade='B'
    else if (averageMarks > 70)
        grade = 'C'
    else
        grade = 'D';
    // Display the result
    document.getElementById('result').innerHTML = `
      <h3>Result</h3>
      <p>Total Marks: ${totalMarks}</p>
      <p>Average Marks: ${averageMarks.toFixed(2)}</p>
      <p>Grade: ${grade}</p>
    `;
  });
  