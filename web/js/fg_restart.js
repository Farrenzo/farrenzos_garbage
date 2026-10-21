const { api } = window.comfyAPI.api;
const { app } = window.comfyAPI.app;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function toast(severity, summary, detail, life = 4000) {
	app.extensionManager?.toast?.add({ severity, summary, detail, life });
}

async function confirmRestart() {
	const dialog = app.extensionManager?.dialog;
	if (dialog?.confirm) {
		return !!(await dialog.confirm({
			title: "Restart ComfyUI",
			message: "Restart the ComfyUI server? Anything currently running will be interrupted.",
			type: "default",
		}));
	}
	return window.confirm("Restart the ComfyUI server? Anything currently running will be interrupted.");
}

async function serverUp() {
	try {
		const r = await api.fetchApi("/system_stats", { cache: "no-store" });
		return r.ok;
	} catch {
		return false;
	}
}

async function restartComfy() {
	if (!(await confirmRestart())) return;

	try {
		await api.fetchApi("/fg/restart", { method: "POST" });
	} catch (e) {
		toast("error", "Restart failed", String(e), 6000);
		return;
	}
	toast("info", "Restarting ComfyUI", "The page will reload when the server is back.", 8000);

	// Wait for the old server to go down (max ~15s), then for the new one to come up.
	for (let i = 0; i < 30 && (await serverUp()); i++) await sleep(500);
	while (!(await serverUp())) await sleep(1000);
	location.reload();
}

app.registerExtension({
	name: "FG.RestartComfy",
	commands: [
		{
			id: "FG.RestartComfy",
			label: "Restart ComfyUI",
			icon: "pi pi-power-off",
			function: restartComfy,
		},
	],
	menuCommands: [{ path: ["Help"], commands: ["FG.RestartComfy"] }],
});