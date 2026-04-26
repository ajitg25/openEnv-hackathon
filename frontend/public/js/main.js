// =====================================================================
// Ambulance Green Corridor — Leaflet map frontend
// =====================================================================

const API = "/api";

// ── Real Bangalore intersection coordinates ──────────────────────────
// Every node is a real named-road intersection from OpenStreetMap.
// Adjacent nodes share actual roads so OSRM routes follow real streets.
const GRID_EASY = [
  [[12.986971,77.590507],[12.986905,77.595077],[12.989189,77.602186],[12.987583,77.610142],[12.988621,77.617302],[12.981633,77.622952]],
  [[12.983119,77.592327],[12.982727,77.597151],[12.982139,77.603496],[12.983053,77.610650],[12.982840,77.615937],[12.975050,77.625296]],
  [[12.972206,77.594011],[12.976700,77.599334],[12.976639,77.601792],[12.974442,77.611112],[12.976008,77.620239],[12.973205,77.622372]],
  [[12.971751,77.594004],[12.971240,77.597745],[12.969419,77.602402],[12.970415,77.610288],[12.971822,77.619495],[12.966878,77.620781]],
  [[12.962489,77.592057],[12.962488,77.594859],[12.965764,77.603879],[12.965069,77.609945],[12.962496,77.618975],[12.962142,77.617813]],
  [[12.959472,77.593499],[12.958366,77.602182],[12.958025,77.605982],[12.958464,77.612100],[12.956583,77.621077],[12.958194,77.614444]],
];
const GRID_MEDIUM = [
  [[12.986971,77.590507],[12.986905,77.595077],[12.989189,77.602186],[12.987273,77.603899],[12.987718,77.609545],[12.988621,77.617302],[12.984467,77.614386],[12.981633,77.622952]],
  [[12.985980,77.589581],[12.983248,77.595747],[12.984754,77.601761],[12.984130,77.604883],[12.984864,77.610497],[12.982840,77.615937],[12.982585,77.615386],[12.980882,77.615930]],
  [[12.983119,77.592327],[12.981473,77.595902],[12.980695,77.597370],[12.980407,77.603862],[12.981036,77.610099],[12.980899,77.615491],[12.976008,77.620239],[12.975050,77.625296]],
  [[12.972206,77.594011],[12.971751,77.594004],[12.976700,77.599334],[12.975185,77.607965],[12.974442,77.611112],[12.975396,77.614389],[12.974718,77.620046],[12.973205,77.622372]],
  [[12.967088,77.594602],[12.971240,77.597745],[12.970337,77.600745],[12.970902,77.604807],[12.970415,77.610288],[12.968479,77.614241],[12.971822,77.619495],[12.972753,77.620182]],
  [[12.962489,77.592057],[12.967325,77.594944],[12.965192,77.599491],[12.965764,77.603879],[12.965069,77.609945],[12.966502,77.613922],[12.966878,77.620781],[12.962496,77.618975]],
  [[12.959472,77.593499],[12.962157,77.594468],[12.960270,77.601409],[12.959934,77.601917],[12.961598,77.612910],[12.961535,77.614764],[12.962142,77.617813],[12.956583,77.621077]],
  [[12.962488,77.594859],[12.958366,77.602182],[12.958025,77.605982],[12.958464,77.612100],[12.958194,77.614444],[12.965422,77.617872],[12.967286,77.611217],[12.972508,77.619099]],
];
const GRID_HARD = [
  [[12.986950,77.590555],[12.987408,77.594794],[12.986577,77.595406],[12.986090,77.598038],[12.984285,77.599872],[12.983857,77.606078],[12.987003,77.612403],[12.985272,77.611749],[12.987877,77.618265],[12.987184,77.618960],[12.987154,77.619952],[12.986417,77.619454]],
  [[12.984750,77.590482],[12.983874,77.594086],[12.985349,77.596320],[12.984099,77.598935],[12.981937,77.604393],[12.982427,77.606048],[12.982413,77.607715],[12.984325,77.611655],[12.983259,77.616550],[12.984905,77.618366],[12.983773,77.617306],[12.981496,77.623045]],
  [[12.983319,77.591534],[12.982884,77.592309],[12.981314,77.596411],[12.982093,77.598844],[12.981235,77.601341],[12.981351,77.605971],[12.981965,77.608996],[12.982647,77.610554],[12.980983,77.611137],[12.983680,77.610969],[12.975668,77.620245],[12.974611,77.620047]],
  [[12.978220,77.590107],[12.979586,77.592704],[12.980456,77.596916],[12.980061,77.597890],[12.979079,77.602479],[12.978387,77.605452],[12.979268,77.607095],[12.979844,77.609581],[12.976205,77.614649],[12.975155,77.618402],[12.973139,77.622407],[12.972644,77.620329]],
  [[12.976467,77.590600],[12.975067,77.593158],[12.976002,77.594656],[12.976447,77.599360],[12.976288,77.602524],[12.975900,77.604288],[12.975006,77.608210],[12.974356,77.611014],[12.975558,77.613316],[12.975195,77.614322],[12.972598,77.618391],[12.972246,77.619479]],
  [[12.974722,77.590382],[12.972321,77.593918],[12.974618,77.596637],[12.972913,77.599222],[12.972859,77.602606],[12.972536,77.603843],[12.972562,77.608328],[12.973347,77.610909],[12.973646,77.613976],[12.972957,77.616753],[12.971382,77.619529],[12.970688,77.618122]],
  [[12.970340,77.591891],[12.970516,77.594259],[12.971101,77.596460],[12.970716,77.599291],[12.969484,77.602485],[12.970446,77.604954],[12.970003,77.606681],[12.969932,77.610482],[12.970469,77.614761],[12.969805,77.616450],[12.966629,77.620473],[12.965810,77.619837]],
  [[12.967732,77.590460],[12.967981,77.592805],[12.966942,77.596509],[12.967232,77.599477],[12.966995,77.601027],[12.967698,77.606343],[12.967163,77.607845],[12.966992,77.611288],[12.966426,77.614222],[12.965173,77.617695],[12.969136,77.615522],[12.968382,77.614321]],
  [[12.962123,77.594460],[12.962799,77.595274],[12.964172,77.596686],[12.964928,77.598278],[12.964584,77.600999],[12.964402,77.606690],[12.965037,77.609900],[12.966897,77.609520],[12.967937,77.611040],[12.968812,77.613372],[12.969179,77.612656],[12.971769,77.614769]],
  [[12.960807,77.594178],[12.962227,77.595965],[12.962116,77.597175],[12.961686,77.598912],[12.961874,77.601535],[12.959561,77.602986],[12.957883,77.606044],[12.966573,77.608765],[12.966397,77.607937],[12.968819,77.610616],[12.969753,77.612111],[12.973341,77.615110]],
  [[12.958018,77.593126],[12.956240,77.592165],[12.961902,77.598043],[12.960401,77.601277],[12.959934,77.601917],[12.958768,77.604579],[12.960948,77.600356],[12.966080,77.606766],[12.965983,77.605815],[12.966956,77.606485],[12.965802,77.604677],[12.971895,77.610576]],
  [[12.955469,77.592053],[12.963579,77.596044],[12.963130,77.601359],[12.964826,77.597232],[12.965241,77.600377],[12.965682,77.603748],[12.965504,77.602281],[12.965500,77.601427],[12.968446,77.606086],[12.969209,77.606424],[12.969439,77.605446],[12.973833,77.612957]],
];

const GRIDS = { 6: GRID_EASY, 8: GRID_MEDIUM, 12: GRID_HARD };

function G(rc) {
  const grid = GRIDS[n];
  if (grid && grid[rc[0]] && grid[rc[0]][rc[1]]) return grid[rc[0]][rc[1]];
  return [12.9716 - rc[0] * 0.004, 77.5946 + rc[1] * 0.005];
}

// ── State ────────────────────────────────────────────────────────────
let obs = null;
let prevAmbLoc = null;  // for smooth animation
let n = 6;
let episodeReward = 0;
let stepNum = 0;
let logLines = [];
let playing = false;
let dispatched = false;

// ── Leaflet map ──────────────────────────────────────────────────────
const map = L.map("map", {
  center: [12.9716, 77.5946], zoom: 15,
  zoomControl: true, attributionControl: false,
});
L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
  maxZoom: 19, subdomains: "abcd",
}).addTo(map);

// Persistent markers — created once, updated in place (no flicker)
let ambMarker = null;     // ambulance
let patientMarker = null;
const hospMarkers = [];   // hospital markers
const hospCircles = [];

// Layers that DO get redrawn each step (cheap)
const cityRoadLayer = L.layerGroup().addTo(map);   // all roads in the city
const citySignalLayer = L.layerGroup().addTo(map);  // all traffic signals
const routeLayer = L.layerGroup().addTo(map);
const signalLayer = L.layerGroup().addTo(map);       // lookahead signals (on top)
const eventLayer = L.layerGroup().addTo(map);

// Store all road line references so we can update colors without redrawing
let _roadLines = {};   // "r1,c1-r2,c2" -> L.polyline
let _signalMarkers = {}; // "r,c" -> L.marker
let _allRoads = [];     // raw road data from server
let _allSignals = {};   // raw signal data from server

// ── Icons ────────────────────────────────────────────────────────────
function emojiIcon(emoji, size, cls) {
  return L.divIcon({
    html: `<div style="font-size:${size}px;text-align:center;line-height:1" class="${cls||''}">${emoji}</div>`,
    iconSize: [size*1.2, size*1.2], iconAnchor: [size*0.6, size*0.6],
    className: "signal-icon",
  });
}

function signalIcon(phase) {
  const nsG = phase === "ns_green";
  const d = (c) => `<div class="signal-dot ${c}" style="width:8px;height:8px"></div>`;
  return L.divIcon({
    html: `<div class="signal-box lookahead">
      <div class="row">${d("dim")}${d(nsG?"green":"red")}${d("dim")}</div>
      <div class="row">${d(nsG?"red":"green")}${d("dim")}${d(nsG?"red":"green")}</div>
      <div class="row">${d("dim")}${d(nsG?"green":"red")}${d("dim")}</div>
    </div>`,
    iconSize: [30,30], iconAnchor: [15,15], className: "signal-icon",
  });
}

function trafficColor(v) {
  v = Math.max(0, Math.min(1, v));
  if (v < 0.3) return "#5a5a70";
  if (v < 0.55) return "#b8942a";
  if (v < 0.75) return "#d4662a";
  return "#e63946";
}
function roadWeight(type) {
  return { highway: 7, main: 5, residential: 3, damaged: 4 }[type] || 3;
}

// ── Smooth ambulance animation ───────────────────────────────────────
function animateMarker(marker, fromLL, toLL, durationMs) {
  const start = performance.now();
  const from = [fromLL[0], fromLL[1]];
  const to = [toLL[0], toLL[1]];

  function step(now) {
    const t = Math.min(1, (now - start) / durationMs);
    // Ease in-out
    const ease = t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t+2, 2)/2;
    const lat = from[0] + (to[0] - from[0]) * ease;
    const lng = from[1] + (to[1] - from[1]) * ease;
    marker.setLatLng([lat, lng]);
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── API ──────────────────────────────────────────────────────────────
async function apiReset() {
  const difficulty = document.getElementById("difficulty").value;
  const res = await fetch(`${API}/reset`, {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ difficulty }),
  });
  return (await res.json()).observation;
}
async function apiStep(action) {
  const res = await fetch(`${API}/step`, {
    method: "POST", headers: {"Content-Type":"application/json"},
    body: JSON.stringify(action),
  });
  const d = await res.json();
  return { obs: d.observation, reward: d.reward ?? 0, done: d.done ?? false };
}
async function apiRoads() {
  const res = await fetch(`${API}/roads`);
  return (await res.json()).roads;
}
async function apiSignals() {
  const res = await fetch(`${API}/signals`);
  return (await res.json()).signals;
}

// ── Policies ─────────────────────────────────────────────────────────
function dispatchAction() {
  const avail = obs.hospitals.filter(h => !h.at_capacity);
  const spec = avail.filter(h => h.specialization === obs.patient_condition);
  const pool = spec.length ? spec : avail.length ? avail : obs.hospitals;
  const best = pool.reduce((a,b) => a.distance_to_patient < b.distance_to_patient ? a : b);
  return { hospital_id: best.hospital_id, signal_controls: [] };
}
function routingAction() {
  const pol = document.getElementById("policy").value;
  const sigs = obs.lookahead_signals || [];
  if (pol === "none") return { signal_controls: [] };
  const controls = [];
  for (const s of sigs) {
    const needed = (s.ambulance_direction==="north"||s.ambulance_direction==="south") ? "ns_green" : "ew_green";
    if (pol === "naive" || (pol === "smart" && s.phase !== needed))
      controls.push({ row: s.row, col: s.col, phase: needed });
  }
  return { signal_controls: controls };
}

// ── DRAW ALL CITY ROADS (called once on reset) ──────────────────────
async function drawCityRoads() {
  cityRoadLayer.clearLayers();
  citySignalLayer.clearLayers();
  _roadLines = {};
  _signalMarkers = {};

  // Fetch all roads and signals from backend
  _allRoads = await apiRoads();
  _allSignals = await apiSignals();

  // Draw every road segment colored by traffic
  for (const road of _allRoads) {
    const a = G(road.from), b = G(road.to);
    const key = `${road.from[0]},${road.from[1]}-${road.to[0]},${road.to[1]}`;
    const color = road.blocked ? "#e63946" : trafficColor(road.traffic);
    const weight = roadWeight(road.type);
    const dashArray = road.type === "damaged" ? "6 4" : null;

    const line = L.polyline([a, b], {
      color, weight, opacity: road.blocked ? 0.7 : 0.55, dashArray,
    }).bindTooltip(
      `<b>${road.type}</b><br>Traffic: ${(road.traffic*100).toFixed(0)}%<br>Quality: ${(road.quality*100).toFixed(0)}%${road.blocked ? "<br><b style='color:red'>BLOCKED</b>" : ""}`,
      { sticky: true }
    ).addTo(cityRoadLayer);

    _roadLines[key] = { line, road };

    // Blocked marker
    if (road.blocked) {
      const mid = [(a[0]+b[0])/2, (a[1]+b[1])/2];
      L.marker(mid, { icon: emojiIcon("🚧", 18), zIndexOffset: 400 }).addTo(cityRoadLayer);
    }
  }

  // Draw all traffic signals (small, dim)
  for (const key in _allSignals) {
    const [r, c] = key.split(",").map(Number);
    const phase = _allSignals[key];
    const nsG = phase === "ns_green";
    const icon = L.divIcon({
      html: `<div style="display:flex;gap:1px"><div style="width:4px;height:4px;border-radius:50%;background:${nsG?"#00E676":"#FF1744"}"></div><div style="width:4px;height:4px;border-radius:50%;background:${nsG?"#FF1744":"#00E676"}"></div></div>`,
      iconSize: [10, 5], iconAnchor: [5, 2.5], className: "signal-icon",
    });
    const m = L.marker(G([r, c]), { icon, zIndexOffset: 100 }).addTo(citySignalLayer);
    _signalMarkers[key] = m;
  }
}

// ── UPDATE CITY ROADS (called each step — updates traffic colors) ────
async function updateCityRoads() {
  const roads = await apiRoads();
  const signals = await apiSignals();

  for (const road of roads) {
    const key = `${road.from[0]},${road.from[1]}-${road.to[0]},${road.to[1]}`;
    const entry = _roadLines[key];
    if (entry) {
      const newColor = road.blocked ? "#e63946" : trafficColor(road.traffic);
      entry.line.setStyle({ color: newColor, opacity: road.blocked ? 0.7 : 0.55 });
      entry.road = road;

      // Add blocked marker if newly blocked
      if (road.blocked && !entry.blockedMarker) {
        const a = G(road.from), b = G(road.to);
        const mid = [(a[0]+b[0])/2, (a[1]+b[1])/2];
        entry.blockedMarker = L.marker(mid, { icon: emojiIcon("🚧", 18), zIndexOffset: 400 }).addTo(cityRoadLayer);
      }
    }
  }

  // Update signal colors
  for (const key in signals) {
    const phase = signals[key];
    const nsG = phase === "ns_green";
    const m = _signalMarkers[key];
    if (m) {
      m.setIcon(L.divIcon({
        html: `<div style="display:flex;gap:1px"><div style="width:4px;height:4px;border-radius:50%;background:${nsG?"#00E676":"#FF1744"}"></div><div style="width:4px;height:4px;border-radius:50%;background:${nsG?"#FF1744":"#00E676"}"></div></div>`,
        iconSize: [10, 5], iconAnchor: [5, 2.5], className: "signal-icon",
      }));
    }
  }
}

// ── FULL RENDER (called once on reset — places everything) ───────────
async function fullRender() {
  if (!obs) return;

  // Clear dynamic layers
  routeLayer.clearLayers();
  signalLayer.clearLayers();
  eventLayer.clearLayers();

  // Remove old persistent markers
  if (ambMarker) { map.removeLayer(ambMarker); ambMarker = null; }
  if (patientMarker) { map.removeLayer(patientMarker); patientMarker = null; }
  hospMarkers.forEach(m => map.removeLayer(m));
  hospMarkers.length = 0;
  hospCircles.forEach(m => map.removeLayer(m));
  hospCircles.length = 0;

  // Draw all city roads + signals
  await drawCityRoads();

  // Hospitals (persistent)
  const HCOLORS = { general:"#CE93D8", cardiac:"#EF5350", trauma:"#FFA726", stroke:"#26C6DA" };
  for (const h of obs.hospitals || []) {
    const isTarget = h.hospital_id === obs.target_hospital_id;
    const color = HCOLORS[h.specialization] || "#CE93D8";
    if (isTarget) {
      const c = L.circle(G(h.location), { radius:120, color, fillColor:color, fillOpacity:0.08, weight:1.5 }).addTo(map);
      hospCircles.push(c);
    }
    const m = L.marker(G(h.location), {
      icon: emojiIcon("🏥", isTarget?28:22), zIndexOffset: isTarget?800:200,
    }).bindTooltip(
      `<b>${h.name}</b>${h.at_capacity?" <span style='color:red'>[FULL]</span>":""}<br>Spec: ${h.specialization}<br>ETA: ${h.travel_time_estimate.toFixed(0)}s`,
      { permanent: isTarget, direction:"top", offset:[0,-14] }
    ).addTo(map);
    hospMarkers.push(m);
  }

  // Patient (persistent)
  patientMarker = L.marker(G(obs.patient_location), {
    icon: emojiIcon("🆘", 26), zIndexOffset: 700,
  }).bindTooltip(`<b>Patient</b><br>${obs.patient_condition}`, {
    permanent: true, direction:"top", offset:[0,-14],
  }).addTo(map);

  // Ambulance (persistent — moved smoothly on step)
  const ambLL = G(obs.ambulance_location);
  ambMarker = L.marker(ambLL, {
    icon: emojiIcon("🚑", 32, "amb-icon"), zIndexOffset: 1000,
  }).addTo(map);
  prevAmbLoc = ambLL;

  // Draw route overlay + lookahead signals + events
  updateDynamicLayers();
}

// ── INCREMENTAL UPDATE (called each step — only updates what changed) ─
function updateDynamicLayers() {
  if (!obs) return;

  routeLayer.clearLayers();
  signalLayer.clearLayers();
  eventLayer.clearLayers();

  // Route segments colored by traffic
  const segs = (obs.current_route && obs.current_route.segments) || [];
  for (const seg of segs) {
    const color = seg.blocked ? "#e63946" : trafficColor(seg.traffic_volume);
    const weight = roadWeight(seg.road_type);
    const dashArray = seg.road_type === "damaged" ? "6 4" : null;
    L.polyline([G(seg.from_pos), G(seg.to_pos)], { color, weight, opacity:0.85, dashArray })
      .bindTooltip(`<b>${seg.road_type}</b><br>Traffic: ${(seg.traffic_volume*100).toFixed(0)}%<br>Quality: ${(seg.road_quality*100).toFixed(0)}%${seg.blocked?"<br><b style='color:red'>BLOCKED</b>":""}`, {sticky:true})
      .addTo(routeLayer);
  }

  // Active route line (glow + solid)
  const routePath = obs.current_route ? obs.current_route.path : [];
  if (routePath.length > 1) {
    const coords = routePath.map(p => G(p));
    L.polyline(coords, { color:"#42A5F5", weight:12, opacity:0.08 }).addTo(routeLayer);
    L.polyline(coords, { color:"#42A5F5", weight:3, opacity:0.8 }).addTo(routeLayer);
  }

  // Alt routes
  for (const alt of obs.alternative_routes || []) {
    if (!alt.path || alt.path.length < 2) continue;
    L.polyline(alt.path.map(p => G(p)), {
      color:"#42A5F5", weight:3, opacity:0.12, dashArray:"8 6",
    }).addTo(routeLayer);
  }

  // Lookahead signals
  for (const la of obs.lookahead_signals || []) {
    L.marker(G([la.row, la.col]), { icon: signalIcon(la.phase), zIndexOffset:500 })
      .bindTooltip(`Signal (${la.row},${la.col})<br>${la.phase==="ns_green"?"N/S: GREEN":"E/W: GREEN"}<br>Traffic: ${(la.traffic_density*100).toFixed(0)}%`)
      .addTo(signalLayer);
  }

  // Events
  for (const evt of obs.active_events || []) {
    const icons = { accident:"⚠️", traffic_spike:"🔥", road_closure:"🚧" };
    L.marker(G(evt.position), { icon: emojiIcon(icons[evt.event_type]||"⚠️",24), zIndexOffset:600 })
      .bindTooltip(evt.description).addTo(eventLayer);
  }

  // Timer bar
  const pct = obs.time_elapsed_seconds / obs.time_limit_seconds;
  const fill = document.getElementById("timeFill");
  fill.style.width = (pct*100)+"%";
  fill.style.background = pct<0.5?"#00E676":pct<0.8?"#FFEA00":"#FF1744";
  document.getElementById("timeLabel").textContent =
    `${obs.time_elapsed_seconds.toFixed(0)}s / ${obs.time_limit_seconds}s  |  Speed: ${(obs.last_speed_factor*100).toFixed(0)}%`;
}

// ── Move ambulance smoothly to new position ──────────────────────────
function moveAmbulance() {
  if (!ambMarker || !obs) return;
  const newLL = G(obs.ambulance_location);
  if (prevAmbLoc && (prevAmbLoc[0] !== newLL[0] || prevAmbLoc[1] !== newLL[1])) {
    animateMarker(ambMarker, prevAmbLoc, newLL, 300);
  }
  prevAmbLoc = newLL;
}

// ── UI updates ───────────────────────────────────────────────────────
function updateUI() {
  document.getElementById("rewardBox").textContent =
    "Reward: " + (episodeReward>=0?"+":"") + episodeReward.toFixed(1);
  const logEl = document.getElementById("log");
  logEl.textContent = logLines.slice(-25).join("\n");
  logEl.scrollTop = logEl.scrollHeight;

  if (!obs) return;
  const pct = ((obs.time_elapsed_seconds/obs.time_limit_seconds)*100).toFixed(0);
  const seg = obs.current_segment;
  const rt = obs.current_route;
  let m = `Phase       : ${dispatched?"routing":"dispatch"}\n`;
  m += `Patient     : ${obs.patient_condition} at (${obs.patient_location})\n`;
  m += `Target      : ${obs.target_hospital_id||"none"}\n\n`;
  m += `Time        : ${obs.time_elapsed_seconds.toFixed(0)}s / ${obs.time_limit_seconds}s (${pct}%)\n`;
  m += `Remaining   : ${obs.intersections_remaining} intersections\n`;
  m += `Speed       : ${(obs.last_speed_factor*100).toFixed(0)}%\n`;
  m += `Red stops   : ${obs.stops_at_red}\n`;
  if (seg) m += `Road        : ${seg.road_type} (q:${(seg.road_quality*100).toFixed(0)}% t:${(seg.traffic_volume*100).toFixed(0)}%)\n`;
  if (rt&&rt.estimated_time>0) m += `Route ETA   : ${rt.estimated_time.toFixed(0)}s\n`;
  m += `\nSignals     : ${obs.necessary_toggles} useful / ${obs.unnecessary_toggles} wasted\n`;
  m += `Efficiency  : ${(obs.signal_efficiency*100).toFixed(0)}%\n`;
  m += `Reroutes    : ${obs.successful_reroutes}\n`;
  if (obs.active_events?.length) {
    m += "\nEvents:\n";
    obs.active_events.forEach(e => m += "  "+e.description+"\n");
  }
  document.getElementById("metrics").textContent = m;
}

function gridSize() {
  return { easy:6, medium:8, hard:12 }[document.getElementById("difficulty").value] || 6;
}

// ── Actions ──────────────────────────────────────────────────────────
async function doReset() {
  playing = false;
  document.getElementById("playBtn").textContent = "\u25B6 Play";
  document.getElementById("metrics").textContent = "Loading...";
  document.getElementById("rewardBox").textContent = "...";
  try { obs = await apiReset(); } catch(err) {
    document.getElementById("metrics").textContent =
      "Connection failed.\n\nStart the servers:\n\n  Terminal 1:\n  cd frontend\n  PYTHONPATH=../envs python env_server.py\n\n  Terminal 2:\n  cd frontend\n  npm start";
    return;
  }
  n = gridSize();
  dispatched = false;
  episodeReward = 0;
  stepNum = 0;
  logLines = [`--- Episode (${document.getElementById("difficulty").value}) ---`,
              `Patient: ${obs.patient_condition} at (${obs.patient_location})`];

  // Fit map
  map.fitBounds([G([n-1,0]), G([0,n-1])], { padding:[30,30] });

  // Auto dispatch
  const action = dispatchAction();
  const result = await apiStep(action);
  obs = result.obs;
  episodeReward += result.reward;
  stepNum++;
  dispatched = true;
  logLines.push("[dispatch] -> " + action.hospital_id);

  updateUI();
  await fullRender();
}

async function doStep() {
  if (!obs || obs.done) { await doReset(); return; }
  const action = dispatched ? routingAction() : dispatchAction();
  if (!dispatched) { dispatched=true; logLines.push("[dispatch] -> "+(action.hospital_id||"?")); }

  const result = await apiStep(action);
  obs = result.obs;
  episodeReward += result.reward;
  stepNum++;

  const ctrls = (action.signal_controls||[]).map(c=>`(${c.row},${c.col})`).join(",")||"none";
  const evts = (obs.active_events||[]).map(e=>e.event_type).join(" ");
  logLines.push(`[${stepNum}] sig:${ctrls} r:${result.reward.toFixed(1)} tot:${episodeReward.toFixed(1)} ${evts}`);

  if (result.done) {
    logLines.push(`--- ${episodeReward>500?"ARRIVED":"TIMED OUT"} | reward:${episodeReward.toFixed(1)} eff:${(obs.signal_efficiency*100).toFixed(0)}% ---`);
  }

  updateUI();
  moveAmbulance();         // smooth animation
  updateDynamicLayers();   // redraw route/signals/events
  updateCityRoads();       // update traffic colors on all roads (async, non-blocking)
}

async function doPlay() {
  if (playing) { playing=false; document.getElementById("playBtn").textContent="\u25B6 Play"; return; }
  playing = true;
  document.getElementById("playBtn").textContent = "\u23F8 Pause";
  if (!obs || obs.done) await doReset();
  while (playing && obs && !obs.done) {
    await doStep();
    await new Promise(r => setTimeout(r, 400));
  }
  playing = false;
  document.getElementById("playBtn").textContent = "\u25B6 Play";
}

// ── Trigger obstacles ────────────────────────────────────────────────
async function doTriggerEvent() {
  if (!obs || !dispatched) return;
  try {
    const res = await fetch(`${API}/trigger_event`, { method: "POST" });
    const data = await res.json();
    if (data.error) { logLines.push(`[event] ${data.error}`); }
    else {
      logLines.push(`[EVENT] ${data.description}`);
      // Refresh roads to show the blocked segment
      await updateCityRoads();
      updateDynamicLayers();
    }
    updateUI();
  } catch(e) { logLines.push(`[event] failed: ${e.message}`); updateUI(); }
}

async function doSpikeTraffic() {
  if (!obs) return;
  try {
    const res = await fetch(`${API}/spike_traffic`, { method: "POST" });
    const data = await res.json();
    logLines.push(`[TRAFFIC] Congestion spike on ${data.spiked}/${data.total} segments`);
    await updateCityRoads();
    updateUI();
  } catch(e) { logLines.push(`[traffic] failed: ${e.message}`); updateUI(); }
}

// ── Wire up buttons ──────────────────────────────────────────────────
document.getElementById("resetBtn").addEventListener("click", doReset);
document.getElementById("stepBtn").addEventListener("click", doStep);
document.getElementById("playBtn").addEventListener("click", doPlay);
document.getElementById("eventBtn").addEventListener("click", doTriggerEvent);
document.getElementById("spikeBtn").addEventListener("click", doSpikeTraffic);
