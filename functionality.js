// Initialize mood data
let moodData = {
    labels: [],
    datasets: [{
        label: 'Mood Rating',
        data: [],
        borderColor: '#7BB5B3',
        tension: 0.4
    }]
};

// Initialize chart
const ctx = document.getElementById('moodChart').getContext('2d');
const moodChart = new Chart(ctx, {
    type: 'line',
    data: moodData,
    options: {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                max: 10
            }
        }
    }
});

function trackMood(rating) {
    const date = new Date().toLocaleDateString();
    moodData.labels.push(date);
    moodData.datasets[0].data.push(rating);

    // Keep only last 7 days
    if (moodData.labels.length > 7) {
        moodData.labels.shift();
        moodData.datasets[0].data.shift();
    }

    moodChart.update();

    // This is where you would send data to your Python AI model
    const entry = {
        date: date,
        mood: rating,
        journal: document.getElementById('journal').value
    };

    // Example endpoint for AI model integration
    // fetch('/api/analyze', {
    //     method: 'POST',
    //     headers: {
    //         'Content-Type': 'application/json',
    //     },
    //     body: JSON.stringify(entry)
    // });
}

function saveEntry() {
    const journal = document.getElementById('journal').value;
    // Here you would typically save to a backend
    alert('Entry saved successfully!');
    document.getElementById('journal').value = '';
}