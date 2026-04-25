const express = require("express");
const { createProxyMiddleware } = require("http-proxy-middleware");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;
const ENV_SERVER = process.env.ENV_SERVER || "http://localhost:8000";

// Proxy /api/* to the OpenEnv Python server
app.use(
  "/api",
  createProxyMiddleware({
    target: ENV_SERVER,
    changeOrigin: true,
    pathRewrite: { "^/api": "" },
  })
);

// Serve static frontend
app.use(express.static(path.join(__dirname, "public")));

app.listen(PORT, () => {
  console.log(`\n  Ambulance Green Corridor`);
  console.log(`  Frontend : http://localhost:${PORT}`);
  console.log(`  Env proxy: ${ENV_SERVER}\n`);
});
