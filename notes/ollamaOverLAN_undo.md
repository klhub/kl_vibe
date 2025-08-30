Got it 👍 You followed the instructions in your file and now want to **undo** those changes. Here’s how to roll them back:

---

### 1. Remove Port Forwarding Rules

Open **PowerShell as Administrator** and run:

```powershell
netsh interface portproxy delete v4tov4 listenport=11434 listenaddress=0.0.0.0
netsh interface portproxy delete v4tov4 listenport=8080 listenaddress=0.0.0.0
```

This removes the forwarding from your Windows host to WSL2.

---

### 2. Remove Firewall Rules

If you created those firewall rules, remove them by running:

```powershell
Remove-NetFirewallRule -DisplayName "Allow Ollama"
Remove-NetFirewallRule -DisplayName "Allow OpenWebUI"
```

---

### 3. Confirm Cleanup

Check that nothing remains:

* Port proxies:

  ```powershell
  netsh interface portproxy show all
  ```

  (Should show nothing now.)

* Firewall rules:

  ```powershell
  Get-NetFirewallRule | Where-Object DisplayName -Match "Ollama|OpenWebUI"
  ```

  (Should return nothing.)

---

👉 After this, your Windows machine will no longer forward ports to WSL2, and the firewall won’t allow inbound traffic on those ports. Effectively, your setup is back to how it was before.

Do you also want me to show you how to **undo it in case you made it persistent across reboots** (e.g., via scheduled tasks or startup scripts)?
