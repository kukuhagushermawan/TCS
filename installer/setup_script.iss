#define MyAppName "TCS - Tree Counting Sawit"
#define MyAppVersion "0.4.0"
#define MyAppPublisher "PT Terramitra Citra Persada"
#define MyAppExeName "TCS.exe"

[Setup]
AppId={{C0919432-9DAF-4519-AF90-1F0867F9B034}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL=https://www.terramitra.com
AppSupportURL=https://www.terramitra.com
AppUpdatesURL=https://www.terramitra.com
DefaultDirName={localappdata}\Programs\TCS
DefaultGroupName=TCS
OutputDir=..\release
OutputBaseFilename=TCS_Setup
SetupIconFile=..\assets\logos\tcs_icon.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=lowest
CloseApplications=yes
LicenseFile=..\legal\EULA_TERMS_B2B.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "..\dist\TCS\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\legal\EULA_TERMS_B2B.txt"; DestDir: "{app}\legal"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\TCS"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\TCS"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall TCS"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Jalankan TCS"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\output"
