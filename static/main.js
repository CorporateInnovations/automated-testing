const fileInput = document.getElementById('file-input');
const uploadBtn = document.getElementById('upload-btn');
const uploadStatus = document.getElementById('upload-status');
const planSection = document.getElementById('plan-section');
const planMeta = document.getElementById('plan-meta');
const testsContainer = document.getElementById('tests-container');
const saveSpecBtn = document.getElementById('save-spec-btn');
const runTestsBtn = document.getElementById('run-tests-btn');
const headedToggle = document.getElementById('headed-toggle');
const previewUrl = document.getElementById('preview-url');
const previewStartBtn = document.getElementById('preview-start-btn');
const previewStopBtn = document.getElementById('preview-stop-btn');
const previewStatus = document.getElementById('preview-status');
const previewImg = document.getElementById('preview-img');
const runSection = document.getElementById('run-section');
const runLog = document.getElementById('run-log');
const runMeta = document.getElementById('run-meta');
let planData = null;
let savedSpecPath = null;
let savedPlanPath = null;
let currentRunId = null;
let pollInterval = null;
let previewSocket = null;
let previewDims = { width: 1280, height: 720 };

uploadBtn.addEventListener('click', async () => {
  if (!fileInput.files[0]) {
    uploadStatus.textContent = 'Pick a file first.';
    return;
  }
  uploadStatus.textContent = 'Uploading and extracting...';
  const fd = new FormData();
  fd.append('file', fileInput.files[0]);

  try {
    const res = await fetch('/extract-tests?include_playwright=true', {
      method: 'POST',
      body: fd,
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || 'Upload failed');
    }
    const data = await res.json();
    planData = data.plan || data; // handle both shapes
    planData.playwright_spec = data.playwright_spec;
    savedPlanPath = data.plan_path || data.latest_plan_path || savedPlanPath;
    savedSpecPath = data.spec_path || data.latest_spec_path || savedSpecPath;
    renderPlan(planData);
    uploadStatus.textContent = 'Extracted successfully.';
    planSection.classList.remove('hidden');
    saveSpecBtn.disabled = false;
    if (savedSpecPath) {
      runTestsBtn.disabled = false;
    }
  } catch (err) {
    uploadStatus.textContent = `Error: ${err.message}`;
  }
});

function renderPlan(plan) {
  const { document_title, summary, test_cases = [] } = plan;
  planMeta.innerHTML = `
    <div><strong>Title:</strong> ${document_title || 'Untitled'}</div>
    <div><strong>Summary:</strong> ${summary || ''}</div>
    <div><strong>Total tests:</strong> ${test_cases.length}</div>
  `;

  const rows = test_cases
    .map(
      (tc) => `
      <tr>
        <td>${tc.id || ''}</td>
        <td>${tc.title || ''}</td>
        <td>${tc.feature || ''}</td>
        <td>${tc.priority || ''}</td>
        <td>${(tc.tags || []).map((t) => `<span class="tag">${t}</span>`).join(' ')}</td>
      </tr>
    `
    )
    .join('');

  testsContainer.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>ID</th><th>Title</th><th>Feature</th><th>Priority</th><th>Tags</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

saveSpecBtn.addEventListener('click', async () => {
  if (!planData || !planData.playwright_spec) return;
  saveSpecBtn.disabled = true;
  saveSpecBtn.textContent = 'Saving...';
  try {
    const res = await fetch('/specs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content: planData.playwright_spec,
        name: planData.document_title || 'plan',
      }),
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      throw new Error(data.error || 'Failed to save spec');
    }
    savedSpecPath = data.path;
    runTestsBtn.disabled = false;
    saveSpecBtn.textContent = 'Spec Saved';
    uploadStatus.textContent = `Spec saved at ${savedSpecPath}`;
  } catch (err) {
    uploadStatus.textContent = `Error: ${err.message}`;
    saveSpecBtn.textContent = 'Save Playwright Spec';
  } finally {
    saveSpecBtn.disabled = false;
  }
});

runTestsBtn.addEventListener('click', async () => {
  if (!savedSpecPath) {
    uploadStatus.textContent = 'Save the spec first.';
    return;
  }
  runTestsBtn.disabled = true;
  runLog.textContent = '';
  runSection.classList.remove('hidden');
  runMeta.textContent = 'Starting run...';

  try {
    const res = await fetch('/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        spec_path: savedSpecPath,
        headed: headedToggle.checked,
      }),
    });
    const data = await res.json();
    if (!res.ok || data.error) {
      throw new Error(data.error || 'Failed to start run');
    }
    currentRunId = data.run_id;
    runMeta.textContent = `Run ${currentRunId} started`;
    pollRun();
  } catch (err) {
    runMeta.textContent = `Error: ${err.message}`;
  } finally {
    runTestsBtn.disabled = false;
  }
});

async function pollRun() {
  if (!currentRunId) return;
  clearInterval(pollInterval);
  pollInterval = setInterval(async () => {
    try {
      const res = await fetch(`/runs/${currentRunId}`);
      const data = await res.json();
      if (!res.ok) {
        runMeta.textContent = 'Run not found';
        clearInterval(pollInterval);
        return;
      }
      renderRun(data);
      if (['succeeded', 'failed'].includes(data.status)) {
        clearInterval(pollInterval);
      }
    } catch (err) {
      runMeta.textContent = `Error polling run: ${err.message}`;
      clearInterval(pollInterval);
    }
  }, 1500);
}

function renderRun(data) {
  runMeta.textContent = `Status: ${data.status} | Started: ${data.started_at || '-'} | Finished: ${data.finished_at || '-'}`;
  const lines = (data.events || []).map((e) => `[${e.timestamp}] ${e.type}: ${e.message}`).join('\n');
  runLog.textContent = lines;
}

function closePreview() {
  if (previewSocket) {
    previewSocket.close();
    previewSocket = null;
  }
  previewStartBtn.disabled = false;
  previewStopBtn.disabled = true;
  previewStatus.textContent = 'Preview stopped';
}

previewStartBtn.addEventListener('click', () => {
  const url = previewUrl.value.trim() || 'https://example.com';
  if (previewSocket) closePreview();
  const wsUrl = `${location.origin.replace(/^http/, 'ws')}/ws/preview?url=${encodeURIComponent(url)}`;
  try {
    previewSocket = new WebSocket(wsUrl);
    previewSocket.binaryType = 'arraybuffer';
    previewStartBtn.disabled = true;
    previewStopBtn.disabled = false;
    previewStatus.textContent = 'Starting preview...';
    previewSocket.onopen = () => {
      previewStatus.textContent = `Streaming ${url}`;
    };
    previewSocket.onmessage = (event) => {
      if (typeof event.data === 'string') {
        if (event.data.startsWith('meta')) {
          const [, w, h] = event.data.split(' ');
          previewDims = { width: parseFloat(w), height: parseFloat(h) };
        } else {
          previewStatus.textContent = event.data;
        }
        return;
      }
      const blob = new Blob([event.data], { type: 'image/jpeg' });
      const objUrl = URL.createObjectURL(blob);
      previewImg.src = objUrl;
    };
    previewSocket.onerror = () => {
      previewStatus.textContent = 'Preview error';
      closePreview();
    };
    previewSocket.onclose = () => {
      closePreview();
    };
  } catch (err) {
    previewStatus.textContent = `Error: ${err.message}`;
    closePreview();
  }
});

previewStopBtn.addEventListener('click', () => {
  closePreview();
});

function sendWheel(evt) {
  if (!previewSocket) return;
  const dx = evt.deltaX;
  const dy = evt.deltaY;
  try {
    previewSocket.send(`wheel ${dx} ${dy}`);
  } catch {
    // ignore
  }
}

previewImg.addEventListener('click', (evt) => {
  if (!previewSocket) return;
  const rect = previewImg.getBoundingClientRect();
  const scaleX = previewDims.width / rect.width;
  const scaleY = previewDims.height / rect.height;
  const x = (evt.clientX - rect.left) * scaleX;
  const y = (evt.clientY - rect.top) * scaleY;
  try {
    previewSocket.send(`click ${x} ${y}`);
  } catch {
    // ignore
  }
});

previewImg.addEventListener('wheel', (evt) => {
  evt.preventDefault();
  sendWheel(evt);
});
