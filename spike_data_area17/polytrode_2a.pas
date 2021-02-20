unit ElectrodeTypes;
interface
uses Windows;
const
  MAXCHANS = 64;
  MAXELECTRODEPOINTS = 12;
  KNOWNELECTRODES = 32;

type
  TElectrode = record
    NumPoints : integer;
    Outline : array[0..MAXELECTRODEPOINTS-1] of TPoint;  //in microns
    NumSites : Integer;
    SiteLoc : array[0..MAXCHANS-1] of TPoint; //in microns
    TopLeftSite, BotRightSite : TPoint;
    CenterX : Integer;
    SiteSize : TPoint; //in microns
    RoundSite : boolean;
    Created : boolean;
    Name : ShortString;
    Description : ShortString;
  end;

function GetElectrode(var Electrode : TElectrode; Name : ShortString) : boolean;

var  KnownElectrode : array[0..KNOWNELECTRODES -1] of TElectrode ;

implementation

procedure MakeKnownElectrodes;
begin
    {µMAP54_2a: this mapping is correct for the cat data
     all dimensions are specified in microns}
    with KnownElectrode[20] do
    begin
      Name := 'µMap54_2a';
      Description := 'µMap54_2a, 65µm spacing';
      SiteSize.x := 15;
      SiteSize.y := 15;
      RoundSite := TRUE;
      Created := FALSE;

      NumPoints := 8;
      Outline[0].x := -100;
      Outline[0].y := 0;
      Outline[1].x := -100;
      Outline[1].y := 1805;
      Outline[2].x := -20;
      Outline[2].y := 2085;
      Outline[3].x := 0;
      Outline[3].y := 2185;
      Outline[4].x := 20;
      Outline[4].y := 2085;
      Outline[5].x := 100;
      Outline[5].y := 1805;
      Outline[6].x := 100;
      Outline[6].y := 0;
      Outline[7].x := Outline[0].x;
      Outline[7].y := Outline[0].y;

      NumSites := 54;
      CenterX := 0;
      SiteLoc[0].x := -28;
      SiteLoc[0].y := 1235;
      SiteLoc[1].x := -28;
      SiteLoc[1].y := 1170;
      SiteLoc[2].x := -28;
      SiteLoc[2].y := 1105;
      SiteLoc[3].x := -28;
      SiteLoc[3].y := 1040;
      SiteLoc[4].x := -28;
      SiteLoc[4].y := 975;
      SiteLoc[5].x := -28;
      SiteLoc[5].y := 910;
      SiteLoc[6].x := -28;
      SiteLoc[6].y := 845;
      SiteLoc[7].x := -28;
      SiteLoc[7].y := 780;
      SiteLoc[8].x := -28;
      SiteLoc[8].y := 715;
      SiteLoc[9].x := -28;
      SiteLoc[9].y := 650;
      SiteLoc[10].x := -28;
      SiteLoc[10].y := 585;
      SiteLoc[11].x := -28;
      SiteLoc[11].y := 520;
      SiteLoc[12].x := -28;
      SiteLoc[12].y := 455;
      SiteLoc[13].x := -28;
      SiteLoc[13].y := 390;
      SiteLoc[14].x := -28;
      SiteLoc[14].y := 325;
      SiteLoc[15].x := -28;
      SiteLoc[15].y := 260;
      SiteLoc[16].x := -28;
      SiteLoc[16].y := 195;
      SiteLoc[17].x := -28;
      SiteLoc[17].y := 130;
      SiteLoc[18].x := -28;
      SiteLoc[18].y := 65;
      SiteLoc[19].x := -28;
      SiteLoc[19].y := 1300;
      SiteLoc[20].x := -28;
      SiteLoc[20].y := 1365;
      SiteLoc[21].x := -28;
      SiteLoc[21].y := 1430;
      SiteLoc[22].x := -28;
      SiteLoc[22].y := 1495;
      SiteLoc[23].x := -28;
      SiteLoc[23].y := 1560;
      SiteLoc[24].x := -28;
      SiteLoc[24].y := 1690;
      SiteLoc[25].x := -28;
      SiteLoc[25].y := 1755;
      SiteLoc[26].x := -28;
      SiteLoc[26].y := 1625;
      SiteLoc[27].x := 28;
      SiteLoc[27].y := 1722;
      SiteLoc[28].x := 28;
      SiteLoc[28].y := 1657;
      SiteLoc[29].x := 28;
      SiteLoc[29].y := 1592;
      SiteLoc[30].x := 28;
      SiteLoc[30].y := 1527;
      SiteLoc[31].x := 28;
      SiteLoc[31].y := 1462;
      SiteLoc[32].x := 28;
      SiteLoc[32].y := 1397;
      SiteLoc[33].x := 28;
      SiteLoc[33].y := 1332;
      SiteLoc[34].x := 28;
      SiteLoc[34].y := 32;
      SiteLoc[35].x := 28;
      SiteLoc[35].y := 97;
      SiteLoc[36].x := 28;
      SiteLoc[36].y := 162;
      SiteLoc[37].x := 28;
      SiteLoc[37].y := 227;
      SiteLoc[38].x := 28;
      SiteLoc[38].y := 292;
      SiteLoc[39].x := 28;
      SiteLoc[39].y := 357;
      SiteLoc[40].x := 28;
      SiteLoc[40].y := 422;
      SiteLoc[41].x := 28;
      SiteLoc[41].y := 487;
      SiteLoc[42].x := 28;
      SiteLoc[42].y := 552;
      SiteLoc[43].x := 28;
      SiteLoc[43].y := 617;
      SiteLoc[44].x := 28;
      SiteLoc[44].y := 682;
      SiteLoc[45].x := 28;
      SiteLoc[45].y := 747;
      SiteLoc[46].x := 28;
      SiteLoc[46].y := 812;
      SiteLoc[47].x := 28;
      SiteLoc[47].y := 877;
      SiteLoc[48].x := 28;
      SiteLoc[48].y := 942;
      SiteLoc[49].x := 28;
      SiteLoc[49].y := 1007;
      SiteLoc[50].x := 28;
      SiteLoc[50].y := 1072;
      SiteLoc[51].x := 28;
      SiteLoc[51].y := 1202;
      SiteLoc[52].x := 28;
      SiteLoc[52].y := 1267;
      SiteLoc[53].x := 28;
      SiteLoc[53].y := 1137;
    end;
  end;

function GetElectrode(var Electrode : TElectrode; Name : ShortString) : boolean;
var i : integer;
begin
   GetElectrode := FALSE;
   For i := 0 to KNOWNELECTRODES-1 do
     if Name = KnownElectrode[i].Name then
     begin
       Move(KnownElectrode[i], Electrode, SizeOf(TElectrode));
       GetElectrode := TRUE;
     end;
end;

initialization
  MakeKnownElectrodes;

end.
