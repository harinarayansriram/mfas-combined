using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

//Dritan Hashorva
namespace FeedForward
{
    internal class Program
    {
        public const String WorkDir = @"c:\feedforward";

        public static readonly int THREAD_NUM = 14;
        public static readonly int MAX_CELL_ID = 136648;
        public static readonly int TOTAL_CONNECTIONS = 41912141;

        public static Dictionary<int, String> CellNames;
        public static Dictionary<String, int> CellNamesR;

        public static Dictionary<long, Connection> ConnectionByCellsIDDict; 

        public static Dictionary<int, List<int>> OutgoingConnections;
        public static Dictionary<int, List<int>> IncomingConnections;

        public static List<List<int>> OutgoingConnections_L1;
        public static List<List<int>> IncomingConnections_L1;

        public static List<int[]> OutgoingConnections_L;
        public static List<int[]> IncomingConnections_L;

        public static long cnt = 0;

        public static readonly object lockObj = new object();

        public static BestSolution bestSolution = new BestSolution();
        public static long lastChangeStep = 0;
        public static long lastStep = 0;

        public static double lastTemperature;

        public static SolutionInstance TheSolution;

        static void Main(string[] args)
        {
            Console.Title = "Feed Forward";

            ReadConnections();
            ReadCellNames();
            //ConvertToCellNames();

            TheSolution = new SolutionInstance(SolutionInstance.GetRandomSolution());
         // TheSolution = SolutionInstance.ReadFromFile($@"{WorkDir}\state\35452425_04065000000.txt", false);
            TheSolution.InstanceID = 1;

            double temperature = 50;
            double coolingRate = 0.000000001;
            cnt = 0;

            while (temperature >= 0)
            {
                ThreadRun(temperature, coolingRate);

                temperature = lastTemperature;

                if (bestSolution.Best != null)
                {
                    String str = $"{DateTime.Now}\t{temperature.ToString("G17")}\t{bestSolution.Score.ToString("00,000,000")}\t{((double)bestSolution.Score) / Program.TOTAL_CONNECTIONS}";
                    Console.WriteLine(str);
                    try
                    {
                        File.AppendAllText($@"{WorkDir}\log.txt", str + Environment.NewLine);
                    }
                    catch { }

                    StringBuilder sb = new StringBuilder();
                    for (int i = 0; i < bestSolution.Best.Count; ++i)
                    {
                        sb.AppendLine($"{bestSolution.Best[i]}");
                    }

                    File.WriteAllText($@"{WorkDir}\state\{bestSolution.Score.ToString("00000000")}_{cnt.ToString("00000000000")}_{temperature.ToString("G17")}.txt", sb.ToString());
                }
            }
        }

        public static void ThreadRun(double temperature, double coolingRate)
        {
            for (int i = 0; i < 1000000; ++i)
            {
                int i1 = TheSolution.random.Next(MAX_CELL_ID);
                int i2 = TheSolution.random.Next(MAX_CELL_ID);

                while (i1 == i2) i2 = TheSolution.random.Next(MAX_CELL_ID);

                TheSolution.Swap(i1, i2, temperature);

                Program.bestSolution.Check(TheSolution);

                ++cnt;

                temperature *= (1 - coolingRate);
            }

            Program.lastTemperature = temperature;
        }

        public static void RandomizeInts(List<int> list)
        {
            Random r = new Random(Guid.NewGuid().GetHashCode());
            for (int i = 0; i < list.Count; ++i)
            {
                int j = r.Next(list.Count);
                (list[i], list[j]) = (list[j], list[i]);
            }
        }

        public static void RandomizeInts(List<int> list, int min_ix, int max_ix)
        {
            Random r = new Random(Guid.NewGuid().GetHashCode());

            int range = max_ix - min_ix + 1;

            for (int i = min_ix; i <= max_ix; ++i)
            {
                int j = min_ix + r.Next(range);
                (list[i], list[j]) = (list[j], list[i]);
            }
        }

        private static void ReadConnections()
        {
            ConnectionByCellsIDDict = new Dictionary<long, Connection>(1 << 19);
            OutgoingConnections = new Dictionary<int, List<int>>();
            IncomingConnections = new Dictionary<int, List<int>>();

            string[] lines = File.ReadAllLines($@"{WorkDir}\graph.csv");

            OutgoingConnections_L1 = new List<List<int>>();
            IncomingConnections_L1 = new List<List<int>>();

            OutgoingConnections_L = new List<int[]>();
            IncomingConnections_L = new List<int[]>();

            for (int i = 0; i <= MAX_CELL_ID; ++i)
            {
                OutgoingConnections_L1.Add(null);
                IncomingConnections_L1.Add(null);
                OutgoingConnections_L.Add(null);
                IncomingConnections_L.Add(null);
            }


            for (int i = 1; i < lines.Length; ++i)
            {
                string[] parts = lines[i].Split(',');

                if (parts.Length != 3) continue;

                Connection c = new Connection() { FromID = Int32.Parse(parts[0]), ToID = Int32.Parse(parts[1]), Weight = Int32.Parse(parts[2]) };

                ConnectionByCellsIDDict.Add(SolutionInstance.GetConnectionHash(c.FromID, c.ToID), c);

                if (OutgoingConnections.ContainsKey(c.FromID))
                {
                    OutgoingConnections[c.FromID].Add(c.ToID);
                }
                else
                {
                    OutgoingConnections.Add(c.FromID, new List<int>() { c.ToID });
                }

                if (IncomingConnections.ContainsKey(c.ToID))
                {
                    IncomingConnections[c.ToID].Add(c.FromID);
                }
                else
                {
                    IncomingConnections.Add(c.ToID, new List<int>() { c.FromID });
                }


                if (OutgoingConnections_L1[c.FromID] == null)
                {
                    OutgoingConnections_L1[c.FromID] = new List<int>();
                }

                if (IncomingConnections_L1[c.ToID] == null)
                {
                    IncomingConnections_L1[c.ToID] = new List<int>();
                }

                OutgoingConnections_L1[c.FromID].Add(c.ToID);
                IncomingConnections_L1[c.ToID].Add(c.FromID);
            }


            for (int i = 0; i <= MAX_CELL_ID; ++i)
            {
                if (OutgoingConnections_L1[i] == null) OutgoingConnections_L[i] = null;
                if (IncomingConnections_L1[i] == null) IncomingConnections_L[i] = null;

                if (OutgoingConnections_L1[i] != null) OutgoingConnections_L[i] = OutgoingConnections_L1[i].ToArray();
                if (IncomingConnections_L1[i] != null) IncomingConnections_L[i] = IncomingConnections_L1[i].ToArray();
            }
        }

        public static void ReadCellNames()
        {
            CellNames = new Dictionary<int, string>();
            CellNamesR = new Dictionary<string, int>();

            String[] lines = File.ReadAllLines($@"{WorkDir}\cells.txt");
            for (int i = 1; i < lines.Length; ++i)
            {
                string[] parts = lines[i].Split(',');
                if (parts.Length == 2)
                {
                    CellNames.Add(Int32.Parse(parts[0]), parts[1]);
                    CellNamesR.Add(parts[1], Int32.Parse(parts[0]));
                }
            }
        }

        public static void ConvertToCellNames()
        {
            String[] lines = File.ReadAllLines($@"{WorkDir}\state\35452425_04065000000.txt");

            StringBuilder sb = new StringBuilder();
            sb.AppendLine("Node ID,Order");

            for (int i = 0; i < lines.Length; ++i)
            {
                if (lines[i].Length == 0) continue;

                string[] parts = lines[i].Split(',');

                sb.AppendLine($"{CellNames[Int32.Parse(parts[0])]},{i}");
            }

            File.WriteAllText($@"{WorkDir}\Result\35452425.csv", sb.ToString());
        }
    }

    public class Connection
    {
        public int FromID { get; set; }
        public int ToID { get; set; }
        public int Weight { get; set; }
    }

    public class Score
    {
        public int Forward { get; set; }
        public int Backward { get; set; }
        public double Ratio { get { return (Forward + Backward) != 0 ? (double)Forward / (Forward + Backward) : 0f; } }
    }

    public class BestSolution
    {
        public List<int> Best;
        public int Score;
        public int InstanceID;

        public bool Check(SolutionInstance si)
        {
            if (si.ForwardScore > Score)
            {
                Score = si.ForwardScore;
                Best = new List<int>(si.Solution);
                InstanceID = si.InstanceID;
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}