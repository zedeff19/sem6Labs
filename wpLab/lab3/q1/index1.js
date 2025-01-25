const questions = [
    {
      question: "What is the capital of France?",
      options: ["Berlin", "Madrid", "Paris", "Rome"],
      correctAnswer: 2 // index of correct answer (0-based)
    },
    {
      question: "Which is the largest planet in our solar system?",
      options: ["Earth", "Jupiter", "Saturn", "Mars"],
      correctAnswer: 1 // index of correct answer (0-based)
    },
    {
      question: "What is the color of the sky on a clear day?",
      options: ["Blue", "Green", "Red", "Yellow"],
      correctAnswer: 0 // index of correct answer (0-based)
    },
    {
      question: "Who wrote 'Hamlet'?",
      options: ["Shakespeare", "Dickens", "Tolkien", "Hemingway"],
      correctAnswer: 0 // index of correct answer (0-based)
    }
  ];
  
  function generateQuiz() {
    const quizContainer = document.getElementById('quiz-container');
    quizContainer.innerHTML = ''; // Clear any existing content
  
    questions.forEach((q, index) => {
      // Create question and options
      const questionDiv = document.createElement('div');
      questionDiv.classList.add('question');
      questionDiv.innerHTML = `<p><strong>${index + 1}. ${q.question}</strong></p>`;
  
      q.options.forEach((option, i) => {
        const optionLabel = document.createElement('label');
        optionLabel.innerHTML = `
          <input type="radio" name="question${index}" value="${i}">
          ${option}
        `;
        questionDiv.appendChild(optionLabel);
        questionDiv.appendChild(document.createElement('br'));
      });
  
      quizContainer.appendChild(questionDiv);
    });
  }
  
  function submitQuiz() {
    let correctAnswers = 0;
  
    questions.forEach((q, index) => {
      const selectedOption = document.querySelector(`input[name="question${index}"]:checked`);
  
      if (selectedOption && parseInt(selectedOption.value) === q.correctAnswer) {
        correctAnswers++;
      }
    });
  
    // Display the result
    const resultElement = document.getElementById('quiz-result');
    resultElement.textContent = `You got ${correctAnswers} out of ${questions.length} correct.`;
  
    if (correctAnswers === questions.length) {
      resultElement.style.color = "green";
    } else {
      resultElement.style.color = "red";
    }
  }
  
  // Generate the quiz when the page loads
  window.onload = generateQuiz;
  