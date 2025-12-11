// Chat functionality
let currentSessionId = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadSuggestedQuestions();
    loadChatHistory();
    setupChatInput();
});

// Chat functionality
function setupChatInput() {
    const input = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    
    if (!input) return;
    
    input.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    input.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });

    // Enable/disable send button based on input
    input.addEventListener('input', function() {
        sendButton.disabled = !this.value.trim();
    });
}

async function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    const sendButton = document.getElementById('sendButton');

    if (!message) return;

    // Disable input while processing
    input.disabled = true;
    sendButton.disabled = true;

    // Add user message to chat
    addMessageToChat('user', message);
    input.value = '';
    input.style.height = 'auto';

    // Show typing indicator
    showTypingIndicator();

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator();
        
        // Add AI response to chat
        addMessageToChat('assistant', data.response);
        
        // Update chat history
        loadChatHistory();

    } catch (error) {
        console.error('Error:', error);
        removeTypingIndicator();
        addMessageToChat('assistant', 'Sorry, I encountered an error while processing your request. Please try again.');
    }

    // Re-enable input
    input.disabled = false;
    sendButton.disabled = true;
    input.focus();
}

function addMessageToChat(type, text) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = message ${type};
    
    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'message-bubble';
    
    // Format the text with line breaks
    const formattedText = text.replace(/\n/g, '<br>');
    bubbleDiv.innerHTML = formattedText;
    
    messageDiv.appendChild(bubbleDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant';
    typingDiv.id = 'typingIndicator';
    
    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'message-bubble';
    bubbleDiv.innerHTML = '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';
    
    typingDiv.appendChild(bubbleDiv);
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Suggested questions
async function loadSuggestedQuestions() {
    try {
        const response = await fetch('/api/suggested-questions');
        const questions = await response.json();
        
        const container = document.getElementById('suggestedQuestions');
        if (!container) return;
        
        container.innerHTML = '';
        
        questions.forEach(question => {
            const button = document.createElement('button');
            button.className = 'question-btn';
            button.textContent = question;
            button.onclick = () => useSuggestedQuestion(question);
            container.appendChild(button);
        });
    } catch (error) {
        console.error('Error loading suggested questions:', error);
    }
}

function useSuggestedQuestion(question) {
    const input = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    
    if (input && sendButton) {
        input.value = question;
        input.focus();
        sendButton.disabled = false;
    }
}

// Chat history
async function loadChatHistory() {
    try {
        const response = await fetch('/api/history');
        const history = await response.json();
        
        const container = document.getElementById('chatHistory');
        if (!container) return;
        
        container.innerHTML = '';
        
        if (history.length === 0) {
            container.innerHTML = '<p style="color: #718096; font-style: italic; text-align: center;">No chat history yet</p>';
            return;
        }
        
        // Only show user messages in history
        const userMessages = history.filter(msg => msg.type === 'user');
        
        userMessages.forEach(msg => {
            const div = document.createElement('div');
            div.style.padding = '8px';
            div.style.borderBottom = '1px solid #e2e8f0';
            div.style.cursor = 'pointer';
            div.textContent = msg.text.length > 50 ? msg.text.substring(0, 50) + '...' : msg.text;
            div.onclick = () => loadMessageFromHistory(msg.text);
            container.appendChild(div);
        });
    } catch (error) {
        console.error('Error loading chat history:', error);
    }
}

function loadMessageFromHistory(message) {
    const input = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    
    if (input && sendButton) {
        input.value = message;
        input.focus();
        sendButton.disabled = false;
    }
}

// User profile
async function saveUserProfile() {
    const background = document.getElementById('educationalBackground').value;
    const interests = document.getElementById('userInterests').value.split(',').map(i => i.trim()).filter(i => i);
    
    try {
        const response = await fetch('/api/save-profile', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                educational_background: background,
                interests: interests
            })
        });
        
        if (response.ok) {
            alert('Profile saved successfully!');
        } else {
            throw new Error('Failed to save profile');
        }
    } catch (error) {
        console.error('Error saving profile:', error);
        alert('Error saving profile. Please try again.');
    }
}

// Utility functions
function loadResource(type) {
    const resources = {
        career_guide: 'Career Guide',
        trends: 'Industry Trends 2024',
        universities: 'Top Universities'
    };
    
    alert(Downloading ${resources[type]}... This feature will be implemented soon!);
}

// Landing page animations
function animateFloatingCards() {
    const cards = document.querySelectorAll('.floating-card');
    cards.forEach((card, index) => {
        card.style.animationDelay = ${index * 2}s;
    });
}

// Smooth scrolling for landing page
function initSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Initialize landing page features
if (document.querySelector('.landing-page')) {
    document.addEventListener('DOMContentLoaded', function() {
        animateFloatingCards();
        initSmoothScrolling();
    });
}