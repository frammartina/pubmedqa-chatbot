document.addEventListener("DOMContentLoaded", () => {
  const recordBtn = document.getElementById("start-recording");
  const sendBtn = document.getElementById("send-btn");
  const userInput = document.getElementById("question");
  const chatLog = document.getElementById("chat-log");

  let recognition;
  let isRecording = false;

  // Funzione per aggiungere messaggi al log
  function addMessage(text, sender) {
    const message = document.createElement("div");
    message.classList.add("message", sender);
    message.textContent = text;
    chatLog.appendChild(message);
    chatLog.scrollTop = chatLog.scrollHeight; // Auto-scroll
    // Scroll con animazione
  chatLog.scrollTo({
    top: chatLog.scrollHeight,
    behavior: "smooth"
  });
  }

  // Evento: invio input testuale
  sendBtn.addEventListener("click", () => {
    const text = userInput.value.trim();
    if (text) {
      addMessage(text, "user");
      userInput.value = ""; // Pulisce l'input
      fetchResponse(text);
    }
  });

  // Evento: start/stop registrazione vocale
  recordBtn.addEventListener("click", () => {
    if (!('webkitSpeechRecognition' in window)) {
      addMessage("Riconoscimento vocale non supportato nel browser.", "bot");
      return;
    }

    if (!isRecording) {
      recognition = new webkitSpeechRecognition();
      recognition.lang = "en-US";
      recognition.continuous = false;
      recognition.interimResults = false;

      // Aggiungo classe animazione al bottone
      recordBtn.classList.add("recording");

  
      recognition.onstart = () => {
        isRecording = true;
        recordBtn.textContent = "Stop Recording";
        addMessage("Recording question...", "bot");
      };

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        addMessage(transcript, "user");
        fetchResponse(transcript);
      };

      recognition.onerror = (event) => {
        addMessage("Errore nel riconoscimento vocale: " + event.error, "bot");
      };

      recognition.onend = () => {
        isRecording = false;
        recordBtn.textContent = "Start Recording";
        addMessage("Recording stopped.", "bot");
         // Rimuovo animazione quando finisce la registrazione
        recordBtn.classList.remove("recording");
      };
     

      recognition.start();

    } else {
      // Stop registrazione se è già in corso
      recognition.stop();
    }
  });

  // Funzione per inviare la domanda al backend
 // Funzione per inviare la domanda al backend e leggere la risposta
async function fetchResponse(question) {
  addMessage("I'm thinking, be patient...", "bot");

  const context = "Fornisci il contesto dinamico o fisso qui";
  const longAnswer = "Fornisci la risposta lunga dinamica o fissa qui";

  try {
    const response = await fetch(" https://f9c1-34-148-66-11.ngrok-free.app/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        question: question,
        context: context,
        long_answer: longAnswer
      })
    });

    if (response.ok) {
      const data = await response.json();
      addMessage(data.answer, "bot");
      // --- Sintesi vocale della risposta ---
      speakText(data.answer);
    } else {
      addMessage("Spiacente, non sono riuscito a processare la tua domanda.", "bot");
    }
  } catch (error) {
    addMessage("Si è verificato un errore durante la connessione al server.", "bot");
    console.error(error);
  }
}

// Funzione per leggere il testo con sintesi vocale
function speakText(text) {
  if ('speechSynthesis' in window) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US"; // Cambia lingua se necessario
    window.speechSynthesis.cancel(); // Ferma eventuali letture precedenti
    window.speechSynthesis.speak(utterance);
  } else {
    alert("La sintesi vocale non è supportata da questo browser.");
  }
}
  
});
