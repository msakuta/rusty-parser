import { entry, parse_ast } from "../pkg/index.js";


async function runCommon(process) {
    // Clear output
    const output = document.getElementById("output");
    output.value = "";
    const canvas = document.getElementById("canvas");
    const canvasRect = canvas.getBoundingClientRect();
    canvas.getContext("2d").clearRect(0, 0, canvasRect.width, canvasRect.height);

    const source = document.getElementById("input").value;
    try{
        process(source);
    }
    catch(e){
        output.value = e;
    }
}

document.getElementById("run").addEventListener("click", () => runCommon(entry));
document.getElementById("parseAst").addEventListener("click", () => runCommon(source => {
    const result = parse_ast(source);
    document.getElementById("output").value = result;
}));

document.getElementById("input").value = `
fn fact(n) {
    if n < 1 {
        1
    } else {
        n * fact(n - 1)
    }
}

print(fact(10));
`;

const samples = document.getElementById("samples");

["expr.dragon", "factorial.dragon", "fibonacci.dragon", "recurse.dragon", "mandel.dragon",
"mandel_canvas.dragon", "str.dragon", "type.dragon", "sieve.dragon",
    "if.dragon", "for.dragon", "fn.dragon", "array.dragon", "array_reverse.dragon", "canvas.dragon"]
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
