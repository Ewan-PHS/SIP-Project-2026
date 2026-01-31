program SIP_Project;

uses
  Vcl.Forms,
  SIP_GUI in 'SIP_GUI.pas' {frmMain};

{$R *.res}

begin
  Application.Initialize;
  Application.MainFormOnTaskbar := True;
  Application.CreateForm(TfrmMain, frmMain);
  Application.Run;
end.
