Prompt : I'm running Ollama in WSL2 together with OpenWebUI. How to access them from another computer?

To access Ollama and OpenWebUI running inside WSL2 from another computer on your local network, you'll need to set up **port forwarding** and adjust your **firewall rules**. Here's a step-by-step guide to get you going:

---

### 🛠️ Step 1: Get WSL2's IP Address
Inside your WSL2 terminal, run:
```bash
ip addr show eth0
```
Look for the `inet` address under `eth0` — that’s your WSL2 IP (e.g., `172.30.104.21`).

---

### 🔁 Step 2: Set Up Port Forwarding on Windows
Open **PowerShell as Administrator** and run the following commands to forward ports from your Windows host to WSL2:

```powershell
netsh interface portproxy add v4tov4 listenport=11434 listenaddress=0.0.0.0 connectport=11434 connectaddress=<WSL2_IP>
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=<WSL2_IP>
```

Replace `<WSL2_IP>` with the IP you found in Step 1.

---

### 🔓 Step 3: Allow Inbound Traffic Through Windows Firewall
Still in PowerShell (Admin), run:

```powershell
New-NetFirewallRule -DisplayName "Allow Ollama" -Direction Inbound -LocalPort 11434 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Allow OpenWebUI" -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow
```

---

### 🌐 Step 4: Access from Another Computer
Now, from another device on the same network, open a browser and go to:

- **OpenWebUI**: `http://<Windows_Host_IP>:8080`
- **Ollama API**: `http://<Windows_Host_IP>:11434`

You can find your Windows host IP by running `ipconfig` in Command Prompt and looking under your active network adapter.

---

Let me know if you want to make this setup persistent across reboots or if you're using Docker inside WSL2 — that adds a few extra twists!