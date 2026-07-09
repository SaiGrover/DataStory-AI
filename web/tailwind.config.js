/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        sage: "#71895B",
        moss: "#8CA66F",
        milk: "#FAFAF7",
        ink: "#15191C",
        line: "#E8E2D7",
      },
      boxShadow: {
        soft: "0 18px 48px rgba(39,43,34,.08)",
      },
      fontFamily: {
        serif: ["Segoe UI Variable Display", "Segoe UI", "system-ui", "sans-serif"],
        sans: ["Inter", "Segoe UI Variable Text", "Segoe UI", "system-ui", "-apple-system", "BlinkMacSystemFont", "Roboto", "Arial", "sans-serif"],
      },
    },
  },
  plugins: [],
};
