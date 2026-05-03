/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    // Forward /api/* to the FastAPI backend in dev so the frontend can use
    // same-origin paths and the WS path works without manual config.
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8000/api/:path*",
      },
    ];
  },
};

module.exports = nextConfig;
