-- 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
--This is Divider IP VHDL RTL
Library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity process_features_sdiv_49s_33s_49_15_1 is
   generic (
       ID : integer :=1;
       NUM_STAGE : integer :=1;
       din0_WIDTH : integer :=16;
       din1_WIDTH : integer :=16;
       dout_WIDTH : integer :=16
   );
   port (

       clk : in std_logic;
       reset : in std_logic;
       ce : in std_logic;

       din0 : in std_logic_vector(din0_WIDTH-1 downto 0);
       din1 : in std_logic_vector(din1_WIDTH-1 downto 0);
       dout : out std_logic_vector(dout_WIDTH-1 downto 0)
   );

function calcDataWidth (x : INTEGER) return INTEGER is
   variable y : INTEGER;
begin
   y := 8;
   while y < x loop
       y := y + 8;
   end loop;
   return y;
end function calcDataWidth;

end entity;

architecture arch of process_features_sdiv_49s_33s_49_15_1 is

   constant dividendUpper : integer := calcDataWidth(din0_WIDTH);
   constant divisorUpper  : integer := calcDataWidth(din1_WIDTH);
   constant IPOUTPUT : integer := dividendUpper + divisorUpper;

   component process_features_sdiv_49s_33s_49_15_1_ip is
       port (

           aclk : in std_logic;
           aclken : in std_logic;

           s_axis_dividend_tvalid : in std_logic;
           s_axis_dividend_tdata : in std_logic_vector(dividendUpper-1 downto 0);
           s_axis_divisor_tvalid : in std_logic;
           s_axis_divisor_tdata : in std_logic_vector(divisorUpper-1 downto 0);
           m_axis_dout_tvalid : out std_logic;
           m_axis_dout_tdata : out std_logic_vector(IPOUTPUT-1 downto 0)
       );
   end component;
   signal IPOut : std_logic_vector(IPOUTPUT-1 downto 0);
   signal IPDividend : std_logic_vector(dividendUpper-1 downto 0);
   signal IPDivisor : std_logic_vector(divisorUpper-1 downto 0);
   signal IPDividend_valid : std_logic;
   signal IPDivisor_valid : std_logic;
   signal IPOut_valid : std_logic;
   signal realQuotient : std_logic_vector(din0_Width-1 downto 0);
   signal realRemainder : std_logic_vector(din1_Width-1 downto 0);

begin
   process_features_sdiv_49s_33s_49_15_1_ip_u : component process_features_sdiv_49s_33s_49_15_1_ip
   port map (

       aclk => clk,
       aclken => ce,

       s_axis_dividend_tvalid => IPDividend_valid,
       s_axis_dividend_tdata  => IPDividend,
       s_axis_divisor_tvalid => IPDivisor_valid,
       s_axis_divisor_tdata  => IPDivisor,
       m_axis_dout_tvalid => IPOut_valid,
       m_axis_dout_tdata => IPOut
   );

   IPDividend_valid <= '1';
   IPDivisor_valid <= '1';
   IPDividend <= std_logic_vector(resize(signed(din0),IPDividend'length));
   IPDivisor <= std_logic_vector(resize(signed(din1),IPDivisor'length));
   realRemainder <= IPOut(din1_WIDTH-1 downto 0);
   realQuotient  <= IPOut(divisorUpper+din0_WIDTH-1 downto divisorUpper);
dout <= std_logic_vector(resize(signed(realQuotient),dout'length));

end architecture;
