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
    const samples = document.getElementById("samples");

    ["expr.dragon", "factorial.dragon", "fibonacci.dragon", "recurse.dragon", "mandel.dragon",
     "str.dragon", "type.dragon", "sieve.dragon",
     "if.dragon", "for.dragon", "fn.dragon", "array.dragon", "array_reverse.dragon"]
     .forEach(fileName => {
        const link = document.createElement("a");
        link.href = "#";
        link.addEventListener("click", () => {
            fetch("scripts/" + fileName)
                .then(file => file.text())
                .then(text => document.getElementById("input").value = text);
        });
        link.innerHTML = fileName;
        samples.appendChild(link);
        samples.append(" ");
    })
}
