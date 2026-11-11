Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = dir
pyw = dir & "\.venv\Scripts\pythonw.exe"
If fso.FileExists(pyw) Then
    shell.Run """" & pyw & """ -X utf8 launcher.py", 0, False
Else
    shell.Run "pythonw -X utf8 launcher.py", 0, False
End If
