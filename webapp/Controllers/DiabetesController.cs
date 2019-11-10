using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using webapp.Models;

namespace webapp.Controllers
{
    public class DiabetesController : Controller
    {
        private readonly ILogger<DiabetesController> _logger;

        public DiabetesController(ILogger<DiabetesController> logger)
        {
            _logger = logger;
        }

        public IActionResult Index()
        {
            // create a viewmodel with default values
            var model = new DiabetesViewModel() 
            {
                NumPregnancies = 0,
                Glucose = 137,
                Diastolic = 40,
                Thickness = 35,
                Insulin = 168,
                BMI = 43.1,
                Pedigree = 2.288,
                Age = 33
            };
            TempData["_statusMessage"] = null;
            
            return View(model);
        }

        [HttpPost]
        public IActionResult Predict(DiabetesViewModel model)
        {
            // TODO: call API
            var result = false;
            TempData["_statusMessage"] = new StatusMessageViewModel() 
            { 
                Message = $"This patient has a {(result ? "high" : "low")} chance of developing diabetes.",
                Type = result ? StatusType.danger : StatusType.success,
                Title = "Result"
            };

            return View("Index", model);
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}
