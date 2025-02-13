// Load and process the data
let topicsData = null;
let formatsData = null;
let statisticsData = null;
let selectedTopic = null;
let selectedFormat = null;
let currentExampleIndex = 0;
let currentExamples = [];
let originalTopicValues = null;
let originalFormatValues = null;

// Fetch all required data
async function loadData() {
    // Wait for Plotly to be available
    if (typeof Plotly === 'undefined') {
        console.log('Waiting for Plotly to load...');
        await new Promise(resolve => setTimeout(resolve, 100));
        return loadData();
    }

    const [topics, formats, statistics] = await Promise.all([
        fetch('assets/data/topics.json').then(r => r.json()),
        fetch('assets/data/formats.json').then(r => r.json()),
        fetch('assets/data/statistics.json').then(r => r.json())
    ]);

    topicsData = topics;
    formatsData = formats;
    statisticsData = statistics;

    // Calculate marginal distributions
    originalTopicValues = new Array(topicsData.length).fill(0);
    originalFormatValues = new Array(formatsData.length).fill(0);

    for (const stat of statistics) {
        originalTopicValues[stat.topic_id] += stat.weight;
        originalFormatValues[stat.format_id] += stat.weight;
    }

    // Create and render initial treemaps
    renderTreemaps(originalTopicValues, originalFormatValues);

    // Initialize domain descriptions
    updateDomainDescriptions();
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

function createTreemapTrace(data, values, type) {
    // Calculate max value for scaling
    const maxValue = Math.max(...values);
    const minFontSize = 10;
    const maxFontSize = 16;
    
    return {
        type: 'treemap',
        labels: data.map((d, i) => {
            const percentage = values[i] * 100;
            return percentage >= 2 ? 
                `${d.domain_name}<br>${percentage.toFixed(1)}%` :
                d.domain_name;
        }),
        parents: new Array(data.length).fill(''),
        values: values,
        textinfo: 'label',
        textposition: 'middle center',
        textfont: { 
            color: 'black',
            // Scale font size based on the value's proportion of the maximum
            size: minFontSize + ((maxFontSize - minFontSize) * (values[0] / maxValue))
        },
        hovertext: data.map((d, i) => d.domain_name + '<br>' + (values[i]*100).toFixed(1) + '%'),
        hoverinfo: 'text',
        hoverlabel: {
            font: { color: 'black' }
        },
        marker: {
            colors: values.map((_, i) =>
                `rgba(${Math.round(data[i].color[0] * 255)},
                      ${Math.round(data[i].color[1] * 255)},
                      ${Math.round(data[i].color[2] * 255)},
                      ${data[i].color[3] * 0.8})`
            )
        },
        tiling: {
            packing: 'squarify',
            flip: "y",
            sort: data.map(d => d.color[1]) // By color gradient
        }
    };
}

function createTreemapLayout() {
    return {
        showlegend: false,
        width: 400,
        height: 400,
        margin: { l: 0, r: 0, t: 0, b: 0 },
        domain: { x: [0, 1], y: [0, 1] }
    };
}

function renderTreemaps(topicValues, formatValues) {
    const config = {
        displayModeBar: true,
        modeBarButtonsToRemove: [
            'toImage', 'sendDataToCloud', 'zoom2d', 'pan2d',
            'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d',
            'autoScale2d', 'resetScale2d'
        ],
        displaylogo: false,
    };

    // Create topic treemap
    const topicTrace = createTreemapTrace(topicsData, topicValues, 'topic');
    const topicLayout = createTreemapLayout();
    Plotly.newPlot('topic-treemap', [topicTrace], topicLayout, config);

    // Create format treemap
    const formatTrace = createTreemapTrace(formatsData, formatValues, 'format');
    const formatLayout = createTreemapLayout();
    Plotly.newPlot('format-treemap', [formatTrace], formatLayout, config);

    // Add hover handlers
    ['topic', 'format'].forEach(type => {
        const plot = document.getElementById(`${type}-treemap`);

        plot.on('plotly_hover', (data) => {
            const point = data.points[0];
            if (point.pointNumber !== undefined) {
                const hoveredId = point.pointNumber;
                updateConditionalDistribution(type, hoveredId);
            }
        });

        plot.on('plotly_unhover', () => {
            // Reset to original distributions when not hovering
            Plotly.update('topic-treemap', {
                values: [originalTopicValues]
            });
            Plotly.update('format-treemap', {
                values: [originalFormatValues]
            });
        });

        // Add click handlers for examples
        plot.on('plotly_treemapclick', (data) => {
            const point = data.points[0];
            if (type === 'topic') {
                selectedTopic = point.pointNumber !== undefined ? point.pointNumber : null;
            } else {
                selectedFormat = point.pointNumber !== undefined ? point.pointNumber : null;
            }
            updateDomainDescriptions();
            loadExamples();
            return false;
        });
    });
}

function updateConditionalDistribution(type, hoveredId) {
    let topicValues = originalTopicValues;
    let formatValues = originalFormatValues;

    if (type === 'format') {
        topicValues = getConditionalDistribution('format', hoveredId);
    } else if (type === 'topic') {
        formatValues = getConditionalDistribution('topic', hoveredId);
    }

    // Update the visualizations with the conditional distributions
    Plotly.update('topic-treemap', {
        values: [topicValues]
    });
    Plotly.update('format-treemap', {
        values: [formatValues]
    });
}

async function loadExamples() {
    let url;
    if (selectedTopic !== null && selectedFormat !== null) {
        url = `assets/data/examples/topic${selectedTopic}_format${selectedFormat}.json`;
    } else if (selectedTopic !== null) {
        url = `assets/data/examples/topic${selectedTopic}.json`;
    } else if (selectedFormat !== null) {
        url = `assets/data/examples/format${selectedFormat}.json`;
    } else {
        currentExamples = [];
        currentExampleIndex = 0;
        updateExampleViewer();
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

function updateExampleViewer() {
    if (!currentExamples || currentExamples.length === 0) {
        document.getElementById('example-viewer').innerHTML = 'Click on a topic or format to view examples!';
        return;
    }

    const example = currentExamples[currentExampleIndex];
    const viewer = document.getElementById('example-viewer');

    // Format the confidence scores as percentages
    const topicScore = Math.round(example.topic_confidence * 100);
    const formatScore = Math.round(example.format_confidence * 100);

    viewer.innerHTML = `
        <div class="example-navigation">
            <button onclick="previousExample()" ${currentExampleIndex === 0 ? 'disabled' : ''}>
                Previous
            </button>
            <span>${currentExampleIndex + 1} / ${currentExamples.length}</span>
            <button onclick="nextExample()" ${currentExampleIndex === currentExamples.length - 1 ? 'disabled' : ''}>
                Next
            </button>
        </div>
        <div class="example-content">
            <div class="example-header">
                <div class="url">${example.url}</div>
                <div class="metadata">
                    Topic: ${topicsData[example.topic_id].domain_name} (<span class="score">${topicScore}%</span>) |
                    Format: ${formatsData[example.format_id].domain_name} (<span class="score">${formatScore}%</span>)
                </div>
            </div>
            <pre>${example.text}</pre>
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

function updateDomainDescriptions() {
    const topicDesc = document.getElementById('topic-description');
    const formatDesc = document.getElementById('format-description');

    // Update topic description
    if (selectedTopic !== null) {
        const topic = topicsData[selectedTopic];
        topicDesc.innerHTML = `
            <h3>${topic.domain_name}</h3>
            <p>${topic.domain_description || 'No description available.'}</p>
        `;
        topicDesc.classList.add('active');
        // Set background color with reduced opacity
        const color = topic.color;
        topicDesc.style.backgroundColor = `rgba(${Math.round(color[0] * 255)}, ${Math.round(color[1] * 255)}, ${Math.round(color[2] * 255)}, 0.2)`;
    } else {
        topicDesc.innerHTML = '<h3 class="default-heading topic-highlight">Topic Domains</h3>';
        topicDesc.classList.add('active');
        topicDesc.style.backgroundColor = 'transparent';
    }

    // Update format description
    if (selectedFormat !== null) {
        const format = formatsData[selectedFormat];
        formatDesc.innerHTML = `
            <h3>${format.domain_name}</h3>
            <p>${format.domain_description || 'No description available.'}</p>
        `;
        formatDesc.classList.add('active');
        // Set background color with reduced opacity
        const color = format.color;
        formatDesc.style.backgroundColor = `rgba(${Math.round(color[0] * 255)}, ${Math.round(color[1] * 255)}, ${Math.round(color[2] * 255)}, 0.2)`;
    } else {
        formatDesc.innerHTML = '<h3 class="default-heading format-highlight">Format Domains</h3>';
        formatDesc.classList.add('active');
        formatDesc.style.backgroundColor = 'transparent';
    }
}

// Initialize on page load with a check for Plotly
function initialize() {
    if (document.readyState === 'complete') {
        loadData();
    } else {
        window.addEventListener('load', loadData);
    }
}

initialize();