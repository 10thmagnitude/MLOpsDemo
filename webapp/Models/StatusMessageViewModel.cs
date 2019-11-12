using System;
using System.ComponentModel.DataAnnotations;

namespace webapp.Models
{
    public enum StatusType 
    {
        success,
        warning,
        info,
        danger
    }

    public class StatusMessageViewModel
    {
        public StatusType Type { get; set; }
        public string Title { get; set; }
        public string Message { get; set; }
    }
}