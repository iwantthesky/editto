#define MyAppName "Editto"
#define MyAppVersion "1.2.0"
#define MyAppPublisher "Engin"
#define MyAppExeName "Editto.exe"

[Setup]
AppId={{7F1C4343-6E58-4FF8-933C-FDBA82B67F23}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\Editto
DefaultGroupName=Editto
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\installer_output
OutputBaseFilename=Editto-Kurulum-1.2.0-Windows-x64
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\Editto.exe
SetupLogging=yes
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"

[Files]
Source: "..\dist\Editto\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\MODEL_LICENSE.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autodesktop}\Editto"; Filename: "{app}\Editto.exe"; Tasks: desktopicon
Name: "{autodesktop}\Editto - Kapat"; Filename: "{app}\Editto.exe"; Parameters: "--stop"; Tasks: desktopicon
Name: "{autodesktop}\Editto - Çıktılar"; Filename: "{app}\Editto.exe"; Parameters: "--open-outputs"; Tasks: desktopicon
Name: "{group}\Editto"; Filename: "{app}\Editto.exe"
Name: "{group}\Editto - Kapat"; Filename: "{app}\Editto.exe"; Parameters: "--stop"
Name: "{group}\Editto - Çıktılar"; Filename: "{app}\Editto.exe"; Parameters: "--open-outputs"
Name: "{group}\Editto'yu Kaldır"; Filename: "{uninstallexe}"

[Tasks]
Name: "desktopicon"; Description: "Masaüstü kısayollarını oluştur"; GroupDescription: "Ek görevler:"; Flags: checkedonce

[Run]
Filename: "{app}\Editto.exe"; Description: "Editto'yu şimdi çalıştır"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{app}\Editto.exe"; Parameters: "--stop"; Flags: runhidden skipifdoesntexist
