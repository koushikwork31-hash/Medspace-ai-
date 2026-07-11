document.addEventListener('DOMContentLoaded', function() {
  const messageInput = document.getElementById('message-input');
  const sendBtn = document.getElementById('send-btn');
  const chatMessages = document.getElementById('chat-messages');

  function sendMessage() {
    const message = messageInput.value.trim();
    if (message === "") return;

    // Append user message
    appendMessage('user', message);
    messageInput.value = '';

    // Send to backend
    fetch('/send_message', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
      appendMessage('bot', data.response);
    })
    .catch(error => {
      appendMessage('bot', 'Sorry, I encountered an error. Please try again later.');
    });
  }

  function appendMessage(sender, text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.textContent = text;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  sendBtn.addEventListener('click', sendMessage);
  messageInput.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
      sendMessage();
    }
  });
});