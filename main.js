import init, { entry } from "./pkg/rusty_parser.js";


async function run() {
    await init();

    // Clear output
    document.getElementById("output").value = "";

    const source = document.getElementById("input").value;
    entry(source);
}

document.getElementById("run").addEventListener("click", run);

// Wait for the html to fully load before loading main.js, since it assumes static html elements exist.
window.onload = () => {
}
