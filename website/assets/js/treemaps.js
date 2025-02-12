// Load and process the data
let topicsData = null;
let formatsData = null;
let statisticsData = null;
let selectedTopic = null;
let selectedFormat = null;
let currentExampleIndex = 0;
let currentExamples = [];

// Fetch all required data
async function loadData() {
    const [topics, formats, statistics] = await Promise.all([
        fetch('assets/topics.json').then(r => r.json()),
        fetch('assets/formats.json').then(r => r.json()),
        fetch('assets/statistics.json').then(r => r.json())
    ]);

    topicsData = topics;
    formatsData = formats;
    statisticsData = statistics;

    // Calculate marginal distributions
    const topicMarginals = new Array(24).fill(0);
    const formatMarginals = new Array(24).fill(0);

    for (const stat of statistics) {
        topicMarginals[stat.topic_id] += stat.weight;
        formatMarginals[stat.format_id] += stat.weight;
    }

    // Create and render initial treemaps
    renderTreemaps(topicMarginals, formatMarginals);
}

function getConditionalDistribution(conditionType, conditionId) {
    if (!conditionType) {
        return null;
    }

    const distribution = new Array(24).fill(0);
    let total = 0;

    for (const stat of statisticsData) {
        if ((conditionType === 'topic' && stat.topic_id === conditionId) ||
            (conditionType === 'format' && stat.format_id === conditionId)) {
            const targetId = conditionType === 'topic' ? stat.format_id : stat.topic_id;
            distribution[targetId] += stat.weight;
            total += stat.weight;
        }
    }

    // Normalize
    if (total > 0) {
        for (let i = 0; i < distribution.length; i++) {
            distribution[i] /= total;
        }
    }

    return distribution;
}

function renderTreemaps(topicValues, formatValues) {
    const topicTrace = {
        type: 'treemap',
        labels: topicsData.map(t => t.domain_name),
        parents: new Array(24).fill(''),
        values: topicValues,
        textinfo: 'label+percent entry',
        marker: {
            colors: topicValues.map((_, i) =>
                `hsl(${350 + i * 5}, 70%, ${60 + (i % 3) * 10}%)`
            )
        },
        hovertemplate: '%{label}<br>%{value:.1%}<br><extra>%{customdata}</extra>',
        customdata: topicsData.map(t => t.domain_description)
    };

    const formatTrace = {
        type: 'treemap',
        labels: formatsData.map(f => f.domain_name),
        parents: new Array(24).fill(''),
        values: formatValues,
        textinfo: 'label+percent entry',
        marker: {
            colors: formatValues.map((_, i) =>
                `hsl(${200 + i * 5}, 70%, ${60 + (i % 3) * 10}%)`
            )
        },
        hovertemplate: '%{label}<br>%{value:.1%}<br><extra>%{customdata}</extra>',
        customdata: formatsData.map(f => f.domain_description)
    };

    const topicLayout = {
        title: 'Topics',
        showlegend: false,
        width: 500,
        height: 500
    };

    const formatLayout = {
        title: 'Formats',
        showlegend: false,
        width: 500,
        height: 500
    };

    Plotly.newPlot('topic-treemap', [topicTrace], topicLayout);
    Plotly.newPlot('format-treemap', [formatTrace], formatLayout);

    // Add click handlers
    document.getElementById('topic-treemap').on('plotly_click', (data) => {
        const point = data.points[0];
        selectedTopic = point.pointNumber;
        updateVisualizations();
        loadExamples();
    });

    document.getElementById('format-treemap').on('plotly_click', (data) => {
        const point = data.points[0];
        selectedFormat = point.pointNumber;
        updateVisualizations();
        loadExamples();
    });
}

async function loadExamples() {
    let url;
    if (selectedTopic !== null && selectedFormat !== null) {
        url = `assets/examples/topic${selectedTopic}_format${selectedFormat}.json`;
    } else if (selectedTopic !== null) {
        url = `assets/examples/topic${selectedTopic}.json`;
    } else if (selectedFormat !== null) {
        url = `assets/examples/format${selectedFormat}.json`;
    } else {
        return;
    }

    try {
        const response = await fetch(url);
        currentExamples = await response.json();
        currentExampleIndex = 0;
        updateExampleViewer();
    } catch (error) {
        console.error('Error loading examples:', error);
    }
}

function updateVisualizations() {
    let topicValues, formatValues;

    if (selectedFormat !== null) {
        topicValues = getConditionalDistribution('format', selectedFormat);
    }

    if (selectedTopic !== null) {
        formatValues = getConditionalDistribution('topic', selectedTopic);
    }

    if (!topicValues) {
        topicValues = statisticsData.reduce((acc, stat) => {
            acc[stat.topic_id] += stat.weight;
            return acc;
        }, new Array(24).fill(0));
    }

    if (!formatValues) {
        formatValues = statisticsData.reduce((acc, stat) => {
            acc[stat.format_id] += stat.weight;
            return acc;
        }, new Array(24).fill(0));
    }

    renderTreemaps(topicValues, formatValues);
}

function updateExampleViewer() {
    if (!currentExamples || currentExamples.length === 0) {
        document.getElementById('example-viewer').innerHTML = 'No examples available';
        return;
    }

    const example = currentExamples[currentExampleIndex];
    const viewer = document.getElementById('example-viewer');
    viewer.innerHTML = `
        <div class="example-content">
            <pre>${example.text}</pre>
        </div>
        <div class="example-navigation">
            <button onclick="previousExample()" ${currentExampleIndex === 0 ? 'disabled' : ''}>
                Previous
            </button>
            <span>${currentExampleIndex + 1} / ${currentExamples.length}</span>
            <button onclick="nextExample()" ${currentExampleIndex === currentExamples.length - 1 ? 'disabled' : ''}>
                Next
            </button>
        </div>
    `;
}

function nextExample() {
    if (currentExampleIndex < currentExamples.length - 1) {
        currentExampleIndex++;
        updateExampleViewer();
    }
}

function previousExample() {
    if (currentExampleIndex > 0) {
        currentExampleIndex--;
        updateExampleViewer();
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', loadData);