' Spustit_Na_Pozadi_Headless.vbs - Tichy spoustec simulace Cyber-Tactics na pozadi
Set objShell = CreateObject("WScript.Shell")
Set objFSO = CreateObject("Scripting.FileSystemObject")

strScriptDir = objFSO.GetParentFolderName(WScript.ScriptFullName)
objShell.CurrentDirectory = strScriptDir

' Detekce pythonw.exe ve virtualnim prostredi .venv nebo v systemu
strVenvPy = strScriptDir & "\.venv\Scripts\pythonw.exe"
strSysPy = "pythonw.exe"

If objFSO.FileExists(strVenvPy) Then
    strPythonExe = """" & strVenvPy & """"
Else
    strPythonExe = strSysPy
End If

' Spusti run_headless.py v kontinuálním režimu na pozadí (okno = 0 -> skryté)
strCmd = strPythonExe & " """ & strScriptDir & "\run_headless.py"" --epochs -1"
objShell.Run strCmd, 0, False
