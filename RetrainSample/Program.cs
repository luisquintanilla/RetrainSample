using System;
using Microsoft.ML;

namespace RetrainSample
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromTextFile<TaxiModel.ModelInput>("taxi-fare-data.csv",hasHeader:true, separatorChar:',');
            var retrainedModel = TaxiModel.RetrainPipeline(mlContext, data);
            mlContext.Model.Save(retrainedModel, data.Schema, "TaxiModel-Retrained.zip");
        }
    }
}
