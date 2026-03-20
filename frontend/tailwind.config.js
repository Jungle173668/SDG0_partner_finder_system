/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // SDGZero brand tokens — matches report_agent.py CSS variables
        navy: {
          DEFAULT: "#0a1f3c",
          light: "#112952",
          subtle: "rgba(10,31,60,0.06)",
        },
        teal: {
          DEFAULT: "#00a896",
          light: "#00c5b0",
          faint: "rgba(0,168,150,0.12)",
        },
        amber: {
          DEFAULT: "#f4a11d",
          light: "#f7b84b",
          faint: "rgba(244,161,29,0.15)",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
