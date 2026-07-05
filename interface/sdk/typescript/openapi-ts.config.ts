import { defineConfig } from "@hey-api/openapi-ts";

export default defineConfig({
  client: "@hey-api/client-fetch",
  input: "../../../knowledge/site/static/openapi.json",
  output: {
    path: "./generated",
    format: "prettier",
  },
  plugins: ["@hey-api/typescript", "@hey-api/sdk"],
});
