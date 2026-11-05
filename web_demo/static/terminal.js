const state = document.querySelector("#state");
const expired = document.querySelector("#expired");
const terminal = new Terminal({
  cursorBlink: true,
  convertEol: true,
  fontFamily: '"DM Mono", "Cascadia Mono", Consolas, monospace',
  fontSize: 13,
  lineHeight: 1.15,
  scrollback: 4000,
  theme: {
    background: "#090a08",
    foreground: "#d7d9cf",
    cursor: "#d8ff43",
    selectionBackground: "#3f4636",
    black: "#171915",
    brightBlack: "#62685d",
    green: "#d8ff43",
    cyan: "#77e2e0",
    red: "#ff7849",
    yellow: "#f0d273"
  }
});
const fitAddon = new FitAddon.FitAddon();
terminal.loadAddon(fitAddon);
terminal.open(document.querySelector("#terminal"));
fitAddon.fit();
terminal.writeln("\x1b[38;2;216;255;67mMyAgent public demo\x1b[0m · 正在创建一次性只读工作区…\r\n");

const configuredBase = (window.MYAGENT_DEMO_BASE || "").replace(/\/$/, "");
const origin = configuredBase || window.location.origin;
const wsBase = origin.replace(/^http/, "ws");
const socket = new WebSocket(`${wsBase}/ws/terminal`);
socket.binaryType = "arraybuffer";

function resize() {
  fitAddon.fit();
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "resize", rows: terminal.rows, cols: terminal.cols }));
  }
}

socket.addEventListener("open", () => {
  state.className = "state connected";
  state.innerHTML = "<i></i> 已连接";
  resize();
  terminal.focus();
});

socket.addEventListener("message", (event) => {
  if (event.data instanceof ArrayBuffer) {
    terminal.write(new Uint8Array(event.data));
    return;
  }
  try {
    const message = JSON.parse(event.data);
    if (message.type === "error") {
      terminal.writeln(`\r\n\x1b[38;2;255;120;73m${message.message}\x1b[0m`);
    }
  } catch {
    terminal.write(event.data);
  }
});

socket.addEventListener("close", () => {
  state.className = "state error";
  state.innerHTML = "<i></i> 已断开";
  expired.hidden = false;
});

socket.addEventListener("error", () => {
  terminal.writeln("\r\n\x1b[38;2;255;120;73m连接失败，请稍后重试。\x1b[0m");
});

terminal.onData((data) => {
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "input", data }));
  }
});

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(resize, 100);
});

const dialog = document.querySelector("#example-dialog");
document.querySelector("#examples").addEventListener("click", () => dialog.showModal());
document.querySelectorAll(".example").forEach((button) => {
  button.addEventListener("click", () => {
    const prompt = `${button.dataset.prompt}\r`;
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "input", data: prompt }));
      terminal.focus();
    }
  });
});
