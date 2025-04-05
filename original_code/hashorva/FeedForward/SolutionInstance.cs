using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace FeedForward
{
    public class SolutionInstance
    {
        public int[] Solution;
        public int[] IXLookup;
        public int ForwardScore;
        public int BackwardScore;
        public Random random;
        public int InstanceID;

        private SolutionInstance()
        {
            random = new Random(Guid.NewGuid().GetHashCode());
        }
        public SolutionInstance(List<int> solution, int forwardScore = 0) : this()
        {
            Solution = solution.ToArray();

            IXLookup = new int[Program.MAX_CELL_ID + 1];

            if (forwardScore == 0)
            {
                Score s = CalculateScore();
                forwardScore = s.Forward;
                BackwardScore = s.Backward;
            }
            else
            {
                for (int i = 0; i < Solution.Length; ++i)
                {
                    IXLookup[Solution[i]] = i;
                }
            }

            ForwardScore = forwardScore;
        }

        public SolutionInstance(BestSolution bs) : this(bs.Best, bs.Score)
        {

        }

        public void Swap(int i1, int i2, double temperature)
        {
            if (i1 == i2) return;

            if (i2 < i1)
            {
                (i1, i2) = (i2, i1);
            }

            int c1 = Solution[i1];
            int c2 = Solution[i2];

            long c1_l = c1;
            long c2_l = c2;

            int[] outgoingFromC1 = Program.OutgoingConnections_L[c1];
            int[] outgoingFromC2 = Program.OutgoingConnections_L[c2];
            int[] incomingToC1 = Program.IncomingConnections_L[c1];
            int[] incomingToC2 = Program.IncomingConnections_L[c2];

            int o_f = 0;

            if (incomingToC1 != null)
            {
                foreach (int c in incomingToC1)
                {
                    int cix = IXLookup[c];

                    if (cix < i1)
                    {
                        long key = ((long)c << 18) | c1; // GetConnectionHash(c, c1);
                        o_f += Program.ConnectionByCellsIDDict[key].Weight;
                    }
                }
            }
            if (incomingToC2 != null)
            {
                foreach (int c in incomingToC2)
                {
                    int cix = IXLookup[c];
                    if (cix < i2)
                    {
                        long key = ((long)c << 18) | c2;
                        o_f += Program.ConnectionByCellsIDDict[key].Weight;
                    }
                }
            }
            if (outgoingFromC1 != null)
            {
                foreach (int c in outgoingFromC1)
                {
                    int cix = IXLookup[c];
                    if (cix == i2) continue;
                    if (cix > i1)
                    {
                        long key = ((long)c1 << 18) | c;
                        o_f += Program.ConnectionByCellsIDDict[key].Weight;
                    }
                }
            }
            if (outgoingFromC2 != null)
            {
                foreach (int c in outgoingFromC2)
                {
                    int cix = IXLookup[c];
                    if (cix == i1) continue;
                    if (cix > i2)
                    {
                        long key = ((long)c2 << 18) | c;
                        o_f += Program.ConnectionByCellsIDDict[key].Weight;
                    }
                }
            }

            IXLookup[Solution[i1]] = i2;
            IXLookup[Solution[i2]] = i1;
            (Solution[i1], Solution[i2]) = (Solution[i2], Solution[i1]);
            (c1, c2) = (c2, c1);
            (outgoingFromC1, outgoingFromC2) = (outgoingFromC2, outgoingFromC1);
            (incomingToC1, incomingToC2) = (incomingToC2, incomingToC1);

            int n_f = 0;

            if (incomingToC1 != null)
            {
                foreach (int c in incomingToC1)
                {
                    int cix = IXLookup[c];

                    if (cix < i1)
                    {
                        long key = ((long)c << 18) | c1;
                        n_f += Program.ConnectionByCellsIDDict[key].Weight;
                    }
                }
            }
            if (incomingToC2 != null)
            {
                foreach (int c in incomingToC2)
                {
                    int cix = IXLookup[c];

                    if (cix < i2)
                    {
                        long key = ((long)c << 18) | c2;
                        n_f += Program.ConnectionByCellsIDDict[key].Weight;
                    }
                }
            }
            if (outgoingFromC1 != null)
            {
                foreach (int c in outgoingFromC1)
                {
                    int cix = IXLookup[c];
                    if (cix == i2) continue;
                    if (cix > i1)
                    {
                        long key = ((long)c1 << 18) | c;
                        n_f += Program.ConnectionByCellsIDDict[key].Weight;
                    }
                }
            }
            if (outgoingFromC2 != null)
            {
                foreach (int c in outgoingFromC2)
                {
                    int cix = IXLookup[c];

                    if (cix > i2)
                    {
                        long key = ((long)c2 << 18) | c;
                        n_f += Program.ConnectionByCellsIDDict[key].Weight;
                    }
                }
            }

            double acceptanceProbability = Math.Exp((n_f - o_f) / temperature);

            if (n_f >= o_f || acceptanceProbability > random.NextDouble())
            {
                ForwardScore += n_f - o_f;
            }
            else
            {
                IXLookup[Solution[i1]] = i2;
                IXLookup[Solution[i2]] = i1;
                (Solution[i1], Solution[i2]) = (Solution[i2], Solution[i1]);
            }
        }

        public Score CalculateScore()
        {
            for (int i = 0; i < Solution.Length; ++i)
            {
                IXLookup[Solution[i]] = i;
            }

            Score score = new Score();

            HashSet<int> visited = new HashSet<int>();

            for (int i = 0; i < Solution.Length; ++i)
            {
                int from = Solution[i];

                if (visited.Contains(from))
                {
                    throw new Exception("Collection corrupted!");
                }
                else
                {
                    visited.Add(from);
                }

                if (!Program.OutgoingConnections.ContainsKey(from)) continue;

                List<int> outgoing = Program.OutgoingConnections[from];

                for (int j = 0; j < outgoing.Count; ++j)
                {
                    int to = outgoing[j];

                    long key = GetConnectionHash(from, to);

                    int to_ix = IXLookup[to];

                    if (to_ix > i)
                    {
                        score.Forward += Program.ConnectionByCellsIDDict[key].Weight;
                    }
                    else
                    {
                        score.Backward += Program.ConnectionByCellsIDDict[key].Weight;
                    }
                }
            }

            return score;
        }

        public static List<int> GetRandomSolution()
        {
            List<int> solution = new List<int>();

            for (int i = 0; i < Program.MAX_CELL_ID; ++i)
            {
                solution.Add(i + 1);
            }

            Program.RandomizeInts(solution);
            return solution;
        }

        public static SolutionInstance ReadFromFile(string file = @"c:\feedforward\benchmark.csv", bool randomize = false)
        {
            List<int> solution = new List<int>();

            string[] lines = File.ReadAllLines(file);

            for (int i = 0; i < lines.Length; ++i)
            {
                string[] parts = lines[i].Split(',');

                if (parts[0].Trim().Length == 0) continue;
                if (!Int32.TryParse(parts[0], out int tmp)) continue;

                solution.Add(tmp);
            }

            if (solution.Count != Program.MAX_CELL_ID)
            {
                throw new Exception("Corrupted solution file!");
            }

            if (randomize)
            {
                Program.RandomizeInts(solution, 0, Program.MAX_CELL_ID-1);
            }


            SolutionInstance si = new SolutionInstance(solution);
            si.IXLookup = new int[Program.MAX_CELL_ID + 1];

            for (int i = 0; i < si.Solution.Length; ++i)
            {
                si.IXLookup[si.Solution[i]] = i;
            }

            return si;
        }

        [MethodImplAttribute(MethodImplOptions.AggressiveInlining)]
        public static long GetConnectionHash(long fromID, long toID)
        {
            return (fromID << 18) | toID;
        }
    }
}