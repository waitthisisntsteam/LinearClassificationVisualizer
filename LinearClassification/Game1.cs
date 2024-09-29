using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using System.Collections.Generic;
using MonoGame.Extended;
using System;
using MonoGame.Extended.Input.InputListeners;
using System.Transactions;

namespace LinearClassification
{
    public class Game1 : Game
    {
        private GraphicsDeviceManager gfx;
        private SpriteBatch spriteBatch;

        private List<Point> DataPoints;
        private List<KeyValuePair<double, double>> CurrentSlopes;
        private List<KeyValuePair<double, double>> CurrentBestSlopes;

        private Perceptron HillClimber;
        double[][] HInputs;
        double[] HDesiredOutputs;
        double HCurrentError;

        private Perceptron LinearClassification;
        double[][] LInputs;
        double[] LDesiredOutputs;
        double LCurrentError;

        public Game1()
        {
            gfx = new GraphicsDeviceManager(this);
            Content.RootDirectory = "Content";
            IsMouseVisible = true;
        }

        static double Error(double output, double desiredOutput) => Math.Abs(desiredOutput - output);

        protected override void Initialize()
        {
            LinearClassification = new Perceptron(2, 0.001, 1, Error);
            LinearClassification.Randomize(new Random(), 0, 1);

            HillClimber = new Perceptron(1, 0.01, 1, Error);
            HillClimber.Randomize(new Random(), 0, 2);

            DataPoints = new List<Point>();
            CurrentSlopes = new List<KeyValuePair<double, double>>();


            DataPoints.Add(new Point(40, 20));
            DataPoints.Add(new Point(20, 30));
            DataPoints.Add(new Point(50, 30));

            DataPoints.Add(new Point(390, 140));
            DataPoints.Add(new Point(380, 120));
            DataPoints.Add(new Point(400, 100));

            DataPoints.Add(new Point(590, 20));
            DataPoints.Add(new Point(640, 30));
            DataPoints.Add(new Point(600, 40));

            CurrentSlopes = new List<KeyValuePair<double, double>>();
            CurrentBestSlopes = new List<KeyValuePair<double, double>>();



            LInputs = new double[DataPoints.Count][];
            for (int i = 0; i < DataPoints.Count; i++) LInputs[i] = [DataPoints[i].X, DataPoints[i].Y];
            LDesiredOutputs = [0, 0, 0, 1, 1, 1, 2, 2, 2];
            LCurrentError = LinearClassification.GetError(LInputs, LDesiredOutputs);


            HInputs = [[40], [20], [50]];
            HDesiredOutputs = [20, 30, 30];
            HCurrentError = HillClimber.GetError(HInputs, HDesiredOutputs);
            CurrentSlopes.Add(new KeyValuePair<double, double>(0, 0));
            CurrentBestSlopes.Add(new KeyValuePair<double, double>(0, 0));

            base.Initialize();
        }

        protected double CalculateY(double m, double x, double b) => m * x + b;

        protected override void LoadContent()
        {
            spriteBatch = new SpriteBatch(GraphicsDevice);
        }

        protected override void Update(GameTime gameTime)
        {
            if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed || Keyboard.GetState().IsKeyDown(Keys.Escape))
                Exit();

            LCurrentError = LinearClassification.TrainLinearClassification(LInputs, LDesiredOutputs, LCurrentError);
            Console.WriteLine(LCurrentError);

            KeyValuePair<double, double> d = new KeyValuePair<double, double>(0, 0);
            HCurrentError = HillClimber.TrainHillClimber(HInputs, HDesiredOutputs, HCurrentError, out d);
            CurrentSlopes[0] = d;
            CurrentBestSlopes[0] = new(HillClimber.Weights[0], HillClimber.Bias);

            base.Update(gameTime);
        }

        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.Clear(Color.White);
            spriteBatch.Begin();

            var d = LinearClassification.Compute(LInputs);
            for (int i = 0; i < DataPoints.Count; i++)
            {
                if (d[i] < 0.5) spriteBatch.DrawPoint(DataPoints[i].X, DataPoints[i].Y, Color.Red, 10);
                else if (d[i] < 1.5) spriteBatch.DrawPoint(DataPoints[i].X, DataPoints[i].Y, Color.Purple, 10);
                else if (d[i] < 2.5) spriteBatch.DrawPoint(DataPoints[i].X, DataPoints[i].Y, Color.Blue, 10);
                else spriteBatch.DrawPoint(DataPoints[i].X, DataPoints[i].Y, Color.Black, 10);
            }

            double y1 = CalculateY(CurrentBestSlopes[0].Key, 0, CurrentBestSlopes[0].Value);
            double y2 = CalculateY(CurrentBestSlopes[0].Key, GraphicsDevice.Viewport.Width, CurrentBestSlopes[0].Value);
            spriteBatch.DrawLine(0, (float)y1, GraphicsDevice.Viewport.Width, (float)y2, Color.Red, 2);

            double ny1 = CalculateY(CurrentSlopes[0].Key, 0, CurrentSlopes[0].Value);
            double ny2 = CalculateY(CurrentSlopes[0].Key, GraphicsDevice.Viewport.Width, CurrentSlopes[0].Value);
            spriteBatch.DrawLine(0, (float)ny1, GraphicsDevice.Viewport.Width, (float)ny2, Color.Pink, 2);

            spriteBatch.End();

            base.Draw(gameTime);
        }
    }
}
