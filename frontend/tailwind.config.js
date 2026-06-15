/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#EEF1F4",
        panel: "#FFFFFF",
        line: "#D8DEE5",
        ink: "#1B2733",
        muted: "#5E6B7A",
        epe: "#2E6F95",
        pfe: "#C0562F",
        cva: "#B8862B",
        gapfill: "rgba(192, 86, 47, 0.12)",
      },
      fontFamily: {
        serif: ["'Source Serif 4'", "serif"],
        sans: ["'IBM Plex Sans'", "ui-sans-serif", "system-ui"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};