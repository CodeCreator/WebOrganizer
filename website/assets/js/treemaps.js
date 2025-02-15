// Load and process the data
let topicsData = null;
let formatsData = null;
let statisticsData = null;
let selectedTopic = null;
let selectedFormat = null;
let currentExampleIndex = 0;
let currentExamples = [];
let topicDistribution = null;
let formatDistribution = null;
let topicIdsToIndices = null;
let formatIdsToIndices = null;

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

    statisticsData = statistics;

    topicIdsToIndices = {
        0:  23 - 3,  // Adult
        1:  23 - 12, // Art & Design
        2:  23 - 20, // Software Dev.
        3:  23 - 18, // Crime & Law
        4:  23 - 17, // Education & Jobs
        5:  23 - 21, // Hardware
        6:  23 - 4,  // Entertainment
        7:  23 - 0,  // Social Life
        8:  23 - 10, // Fashion & Beauty
        9:  23 - 22, // Finance & Business
        10: 23 -  5, // Food & Dining
        11: 23 -  6, // Games
        12: 23 -  13, // Health
        13: 23 -  9, // History
        14: 23 -  7, // Home & Hobbies
        15: 23 -  23, // Industrial
        16: 23 -  14, // Literature
        17: 23 -  1, // Politics
        18: 23 -  2, // Religion
        19: 23 -  19, // Science & Tech.
        20: 23 -  15, // Software
        21: 23 -  11, // Sports & Fitness
        22: 23 -  16, // Transportation
        23: 23 -  8, // Travel
    }

    formatIdsToIndices = {
        0:  23 - 7,   // Academic Writing
        1:  23 - 8,   // Content Listing
        2:  23 - 16,  // Creative Writing
        3:  23 - 5,   // Customer Support Page
        4:  23 - 21,  // Discussion Forum / Comment Section
        5:  23 - 3,   // FAQs
        6:  23 - 13,  // Incomplete Content
        7:  23 - 17,  // Knowledge Article
        8:  23 - 4,   // Legal Notices
        9:  23 - 9,   // Listicle
        10: 23 -  10, // News Article
        11: 23 -  11, // Nonfiction Writing
        12: 23 -  1,  // Organizational About Page
        13: 23 -  2,  // Organizational Announcement
        14: 23 -  22, // Personal About Page
        15: 23 -  23, // Personal Blog
        16: 23 -  0,  // Product Page
        17: 23 -  19, // Q&A Forum
        18: 23 -  12, // Spam / Ads
        19: 23 -  15, // Structured Data
        20: 23 -  6,  // Technical Writing
        21: 23 -  14, // Transcript / Interview
        22: 23 -  18, // Tutorial / How-To Guide
        23: 23 -  20, // User Reviews
    }

    topicsData = (
            topics
                .map(topic => ({...topic, index: topicIdsToIndices[topic.domain_id]}))
                .sort((a, b) => a.index - b.index)
    )
    formatsData = (
        formats
            .map(format => ({...format, index: formatIdsToIndices[format.domain_id]}))
            .sort((a, b) => a.index - b.index)
    )

    // Calculate marginal distributions
    topicDistribution = new Array(topicsData.length).fill(0);
    formatDistribution = new Array(formatsData.length).fill(0);

    for (const stat of statistics) {
        topicDistribution[topicIdsToIndices[stat.topic_id]] += stat.weight;
        formatDistribution[formatIdsToIndices[stat.format_id]] += stat.weight;
    }

    // Normalize distributions by sum and round to 2 decimal places
    const topicSum = topicDistribution.reduce((sum, value) => sum + value, 0);
    const formatSum = formatDistribution.reduce((sum, value) => sum + value, 0);
    topicDistribution = topicDistribution.map(d => Math.round((d / topicSum) * 1000) / 1000);
    formatDistribution = formatDistribution.map(d => Math.round((d / formatSum) * 1000) / 1000);

    // Create and render initial treemaps
    renderTreemaps(topicDistribution, formatDistribution);

    // Initialize domain descriptions
    updateDomainDescriptions();
}

function getConditionalDistribution(conditionType, conditionIndex) {
    if (!conditionType) {
        return null;
    }

    const distribution = new Array(24).fill(0);
    let total = 0;

    for (const stat of statisticsData) {
        if ((conditionType === 'topic' && stat.topic_id === topicsData[conditionIndex].domain_id) ||
            (conditionType === 'format' && stat.format_id === formatsData[conditionIndex].domain_id)) {
            const targetId = conditionType === 'topic' ? stat.format_id : stat.topic_id;
            const targetIndex = conditionType === 'topic' ? formatIdsToIndices[targetId] : topicIdsToIndices[targetId];
            distribution[targetIndex] += stat.weight;
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
    return {
        type: 'treemap',
        parents: new Array(data.length).fill(''),
        textposition: 'middle center',
        textfont: {
            color: 'black',
            size: 10
        },
        hoverinfo: 'text',
        hoverlabel: {
            font: { color: 'black' }
        },
        sort: false,
        tiling: {
            packing: 'squarify',
            flip: 'y',
            pad: 2,
            squarifyratio: 1,
        },
        textinfo: 'label+percent root',
        labels: data.map((d) => d.domain_name),
        values: values,
        marker: {
            colors: values.map((_, i) =>
                `rgba(${Math.round(data[i].color[0] * 255)},
                      ${Math.round(data[i].color[1] * 255)},
                      ${Math.round(data[i].color[2] * 255)},
                      ${data[i].color[3] * 0.8})`
            )
        },
        hovertext: data.map((d, i) => d.domain_name + '<br>' + (values[i]*100).toFixed(1) + '%'),
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

function getFullName(domain) {
    return domain.domain_fullname || domain.domain_name.replace(/<br>/g, " ");
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
                const hoveredIndex = point.pointNumber;
                let topicValues = topicDistribution;
                let formatValues = formatDistribution;

                if (type === 'format') {
                    topicValues = getConditionalDistribution('format', hoveredIndex);
                    Plotly.update('topic-treemap', {
                        values: [topicValues],
                    })
                } else if (type === 'topic') {
                    formatValues = getConditionalDistribution('topic', hoveredIndex);
                    Plotly.update('format-treemap', {
                        values: [formatValues]
                    });
                }
            }
        });

        plot.on('plotly_unhover', () => {
            // Reset to original distributions when not hovering
            Plotly.update('topic-treemap', {
                values: [topicDistribution]
            });
            Plotly.update('format-treemap', {
                values: [formatDistribution]
            });
        });

        // Add click handlers for examples
        plot.on('plotly_treemapclick', (data) => {
            const point = data.points[0];
            if (type === 'topic') {
                selectedTopic = point.pointNumber !== undefined ? topicsData[point.pointNumber].domain_id : null;
            } else {
                selectedFormat = point.pointNumber !== undefined ? formatsData[point.pointNumber].domain_id : null;
            }
            updateDomainDescriptions();
            loadExamples();
            return false;
        });
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

    const topicName = getFullName(topicsData[topicIdsToIndices[example.topic_id]]);
    const formatName = getFullName(formatsData[formatIdsToIndices[example.format_id]]);

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
                    Topic: ${topicName} (<span class="score">${topicScore}%</span>) |
                    Format: ${formatName} (<span class="score">${formatScore}%</span>)
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
        const topic = topicsData[topicIdsToIndices[selectedTopic]];
        topicDesc.innerHTML = `
            <h3>${getFullName(topic)}</h3>
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
        const format = formatsData[formatIdsToIndices[selectedFormat]];
        formatDesc.innerHTML = `
            <h3>${getFullName(format)}</h3>
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