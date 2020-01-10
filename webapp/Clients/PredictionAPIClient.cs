using System.Net.Http;
using System.Threading.Tasks;
using webapp.Models;
using System.Text;
using System.Text.Json;
using System;

public interface IPredictionAPIClient
{
    Task<bool> GetPrediction(DiabetesViewModel model);
}

class PredictionResult 
{
    public int[] result { get; set; } 
}

public class PredictionAPIClient : IPredictionAPIClient
{
    public PredictionAPIClient(HttpClient client)
    {
        Client = client;
    }

    public HttpClient Client { get; }

    public async Task<bool> GetPrediction(DiabetesViewModel model) 
    {
        var json = new {
            data = new[] { model }
        };

        var result = await Client.PostAsync("/score", 
            new StringContent(JsonSerializer.Serialize(json), Encoding.UTF8, "application/json"));
        
        if (result.IsSuccessStatusCode)
        {
            var jsonData = await result.Content.ReadAsStringAsync();
            jsonData = jsonData.Substring(1, jsonData.Length - 2).Replace("\\", "");
            var resultData = JsonSerializer.Deserialize<PredictionResult>(jsonData);
            return resultData.result[0] == 0;
        }
        else
        {
            string content = await result.Content.ReadAsStringAsync().Result;
            throw new ApplicationException($"Error calling prediction API: {content}");
        }
    }
}