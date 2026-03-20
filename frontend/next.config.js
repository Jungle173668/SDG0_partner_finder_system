/** @type {import('next').NextConfig} */
const nextConfig = {
  // Proxy /api/* to FastAPI backend during development
  async rewrites() {
    const backend = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backend}/api/:path*`,
      },
      {
        source: "/static/:path*",
        destination: `${backend}/static/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
