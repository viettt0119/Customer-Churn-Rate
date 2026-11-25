const PREFIX = "churn-radar-";
const CACHE = PREFIX + "{{ version }}";
const V = "?v={{ version }}";
const SHELL = [
  "/",
  "/manifest.webmanifest",
  "/static/css/style.css" + V,
  "/static/js/app.js" + V,
  "/static/vendor/chartjs/chart.umd.min.js" + V,
  "/static/vendor/plotly/plotly-cartesian.min.js" + V,
  "/static/vendor/fontawesome/css/fontawesome.min.css" + V,
  "/static/vendor/fontawesome/css/solid.min.css" + V,
  "/static/vendor/fontawesome/webfonts/fa-solid-900.woff2",
  "/static/icons/icon-192.png",
];
const DATA = ["/api/insights", "/api/models", "/api/options"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE)
      .then((cache) => cache.addAll([...SHELL, ...DATA].map((url) => new Request(url, { cache: "reload" }))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith(PREFIX) && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

async function networkFirst(request, key) {
  const cache = await caches.open(CACHE);
  try {
    const response = await fetch(request);
    if (response.ok) await cache.put(key, response.clone());
    return response;
  } catch (err) {
    const hit = await cache.match(key);
    if (hit) return hit;
    throw err;
  }
}

async function cacheFirst(request) {
  const cache = await caches.open(CACHE);
  return (await cache.match(request)) || fetch(request);
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== self.location.origin) return;
  if (request.mode === "navigate" && url.pathname === "/") {
    event.respondWith(networkFirst(request, "/"));
  } else if (DATA.includes(url.pathname)) {
    event.respondWith(networkFirst(request, url.pathname));
  } else if (url.pathname.startsWith("/static/")) {
    event.respondWith(cacheFirst(request));
  }
});
